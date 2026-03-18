#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-5}"
DATASET="${2:-tiny_imagenet}"
BASE_PATH="${3:-/path/to/gtai/FDS}"
N_CLIENTS="${4:-2}"
SEED="${5:-42}"
WORKLOAD_SCRIPT="${6:-}"

MONITOR_SECONDS="${MONITOR_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"
REBALANCE_AFTER_SECONDS="${REBALANCE_AFTER_SECONDS:-60}"
REBALANCE_COOLDOWN_SECONDS="${REBALANCE_COOLDOWN_SECONDS:-45}"

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
  echo "COUNT must be a positive integer" >&2
  exit 1
fi

REPO_ROOT="/path/to/data/FDS"
cd "$REPO_ROOT"

source scripts/hpc_session_env.sh >/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate new

mkdir -p local_artifacts/experiments/batch local_artifacts/splits/tiny_imagenet

SPLIT_PATH="$REPO_ROOT/local_artifacts/splits/tiny_imagenet/random_n${N_CLIENTS}_seed${SEED}.json"
if [[ ! -f "$SPLIT_PATH" ]]; then
  python scripts/generate_baselines.py \
    --dataset "$DATASET" \
    --base_path "$BASE_PATH" \
    --output "$REPO_ROOT/local_artifacts/splits/tiny_imagenet" \
    --n_clients "$N_CLIENTS" \
    --seed "$SEED"
fi

THIRD_PARTITION=""
if sinfo -h -o "%P" | sed 's/*//g' | grep -qx "gengpu"; then
  THIRD_PARTITION="gengpu"
elif sinfo -h -o "%P" | sed 's/*//g' | grep -qx "gpu-a100"; then
  THIRD_PARTITION="gpu-a100"
fi

PARTITIONS=("gtai" "preemptgpu")
declare -A PARTITION_CAPS=(
  [gtai]=2
  [preemptgpu]=2
)
if [[ -n "$THIRD_PARTITION" ]]; then
  PARTITIONS+=("$THIRD_PARTITION")
  PARTITION_CAPS["$THIRD_PARTITION"]=4
fi

total_capacity=0
for p in "${PARTITIONS[@]}"; do
  total_capacity=$((total_capacity + PARTITION_CAPS[$p]))
done
if [[ "$COUNT" -gt "$total_capacity" ]]; then
  echo "Requested $COUNT jobs exceeds configured partition caps ($total_capacity)." >&2
  exit 1
fi

if [[ -n "$WORKLOAD_SCRIPT" && ! -f "$WORKLOAD_SCRIPT" ]]; then
  echo "WORKLOAD_SCRIPT not found: $WORKLOAD_SCRIPT" >&2
  exit 1
fi

BATCH_ID="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$REPO_ROOT/local_artifacts/experiments/batch/${BATCH_ID}_status.tsv"
: > "$STATUS_FILE"

declare -A ATTEMPT
declare -A JOB_NAME
declare -A JOB_ID
declare -A JOB_PARTITION
declare -A SUBMIT_EPOCH
declare -A LAST_REBALANCE_EPOCH
declare -A ASSIGNED_THIS_BATCH

for p in "${PARTITIONS[@]}"; do
  ASSIGNED_THIS_BATCH[$p]=0
done

echo "batch_id=$BATCH_ID"
echo "status_file=$STATUS_FILE"
echo "session_names=1..$COUNT"

session_for() {
  local idx="$1"
  echo "$idx"
}

gres_for() {
  local partition="$1"
  case "$partition" in
    gtai) echo "gpu:a100_80g:1" ;;
    preemptgpu) echo "gpu:a100_80g:1" ;;
    gpu-a100) echo "gpu:a100:1" ;;
    gengpu) echo "gpu:1" ;;
    *) echo "gpu:1" ;;
  esac
}

job_name_for() {
  local idx="$1"
  local attempt="$2"
  echo "fds_job_${BATCH_ID}_${idx}_a${attempt}"
}

running_count() {
  local partition="$1"
  squeue -u "$USER" -h -p "$partition" -o "%T" | grep -Ec 'RUNNING|CONFIGURING' || true
}

pending_count() {
  local partition="$1"
  squeue -u "$USER" -h -p "$partition" -o "%T" | grep -Ec 'PENDING' || true
}

best_partition_for_launch() {
  local best=""
  local best_score=999999

  for partition in "${PARTITIONS[@]}"; do
    local run pending cap load score
    run="$(running_count "$partition")"
    pending="$(pending_count "$partition")"
    cap="${PARTITION_CAPS[$partition]}"
    load=$((run + pending + ASSIGNED_THIS_BATCH[$partition]))
    if [[ "$load" -ge "$cap" ]]; then
      continue
    fi
    score=$((pending * 10 + run))
    if [[ "$score" -lt "$best_score" ]]; then
      best="$partition"
      best_score="$score"
    fi
  done

  if [[ -n "$best" ]]; then
    echo "$best"
    return
  fi

  for partition in "${PARTITIONS[@]}"; do
    local pending score
    pending="$(pending_count "$partition")"
    score=$((pending * 10))
    if [[ "$score" -lt "$best_score" ]]; then
      best="$partition"
      best_score="$score"
    fi
  done

  echo "$best"
}

best_partition_for_rebalance() {
  local current="$1"
  local best=""
  local best_pending=999999

  for partition in "${PARTITIONS[@]}"; do
    local run pending cap load
    [[ "$partition" == "$current" ]] && continue
    run="$(running_count "$partition")"
    pending="$(pending_count "$partition")"
    cap="${PARTITION_CAPS[$partition]}"
    load=$((run + pending))
    if [[ "$load" -ge "$cap" ]]; then
      continue
    fi
    if [[ "$pending" -lt "$best_pending" ]]; then
      best="$partition"
      best_pending="$pending"
    fi
  done

  echo "$best"
}

default_workload_command() {
  local run_name="$1"
  cat <<EOF
python scripts/preflight_fl.py --config-dir $REPO_ROOT/fl_training/conf --config-name config --override dataset=$DATASET --override model=resnet18 --override model.args.pretrained=true --override dataset.args.root=$BASE_PATH --override dataset.dataset_splits=$SPLIT_PATH --override fed.rounds=1 --override fed.n_clients=$N_CLIENTS --override training.local_steps=1 --override name=$run_name --override paths.output_dir=$REPO_ROOT/local_artifacts/experiments/outputs; python fl_training/run_experiment.py --config-path $REPO_ROOT/fl_training/conf --config-name config dataset=$DATASET model=resnet18 model.args.pretrained=true dataset.args.root=$BASE_PATH dataset.dataset_splits=$SPLIT_PATH fed.rounds=1 fed.n_clients=$N_CLIENTS training.local_steps=1 name=$run_name paths.output_dir=$REPO_ROOT/local_artifacts/experiments/outputs
EOF
}

launch_index() {
  local idx="$1"
  local partition="$2"
  local attempt="${ATTEMPT[$idx]}"
  local session job_name run_name log_file gres workload_cmd

  session="$(session_for "$idx")"
  job_name="$(job_name_for "$idx" "$attempt")"
  run_name="exp_${BATCH_ID}_${idx}_a${attempt}"
  log_file="$REPO_ROOT/local_artifacts/experiments/batch/${BATCH_ID}_job${idx}.log"
  gres="$(gres_for "$partition")"

  tmux kill-session -t "$session" 2>/dev/null || true

  if [[ -n "$WORKLOAD_SCRIPT" ]]; then
    workload_cmd="export EXPERIMENT_INDEX=$idx EXPERIMENT_BATCH_ID=$BATCH_ID EXPERIMENT_RUN_NAME=$run_name EXPERIMENT_DATASET=$DATASET EXPERIMENT_BASE_PATH=$BASE_PATH EXPERIMENT_SPLIT_PATH=$SPLIT_PATH EXPERIMENT_OUTPUT_ROOT=$REPO_ROOT/local_artifacts/experiments/outputs EXPERIMENT_N_CLIENTS=$N_CLIENTS; bash $WORKLOAD_SCRIPT"
  else
    workload_cmd="$(default_workload_command "$run_name")"
  fi

  inner_cmd="cd $REPO_ROOT && source scripts/hpc_session_env.sh >/dev/null && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate new && export tiny_imagenet_chosen_split_path=$SPLIT_PATH && exec > >(tee -a $log_file) 2>&1 && echo [JOB-START] \$(date -Is) idx=$idx part=$partition name=$job_name && srun -p $partition --job-name=$job_name --gres=$gres --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash -lc 'set -e; cd $REPO_ROOT; source scripts/hpc_session_env.sh >/dev/null; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate new; export tiny_imagenet_chosen_split_path=$SPLIT_PATH; $workload_cmd' ; rc=\$?; echo [JOB-END] \$(date -Is) rc=\$rc; exec bash -l"

  tmux new-session -d -s "$session" "$inner_cmd"

  JOB_NAME[$idx]="$job_name"
  JOB_ID[$idx]=""
  JOB_PARTITION[$idx]="$partition"
  SUBMIT_EPOCH[$idx]="$(date +%s)"
  LAST_REBALANCE_EPOCH[$idx]="0"

  echo -e "${idx}\t${session}\t${job_name}\t${partition}\tSUBMITTED\t-\t-" | tee -a "$STATUS_FILE"
}

for ((i=1; i<=COUNT; i++)); do
  ATTEMPT[$i]=1
  partition="$(best_partition_for_launch)"
  if [[ -z "$partition" ]]; then
    echo "No usable partition found for job index $i" >&2
    exit 1
  fi
  launch_index "$i" "$partition"
  ASSIGNED_THIS_BATCH[$partition]=$((ASSIGNED_THIS_BATCH[$partition] + 1))
done

DISCOVERED_COUNT=0
discover_job_ids() {
  local discovered=0
  for ((i=1; i<=COUNT; i++)); do
    local job_name line jid state reason state_reason
    job_name="${JOB_NAME[$i]}"
    line="$(squeue -u "$USER" -h -n "$job_name" -o "%A|%T|%R" | head -n 1 || true)"
    if [[ -n "$line" ]]; then
      jid="${line%%|*}"
      state_reason="${line#*|}"
      state="${state_reason%%|*}"
      reason="${state_reason#*|}"
      if [[ "${JOB_ID[$i]:-}" != "$jid" ]]; then
        JOB_ID[$i]="$jid"
        echo -e "${i}\t$(session_for "$i")\t${job_name}\t${JOB_PARTITION[$i]}\t${state}\t${jid}\t${reason}" | tee -a "$STATUS_FILE"
      fi
      discovered=$((discovered + 1))
    fi
  done
  DISCOVERED_COUNT="$discovered"
}

echo "Discovering job IDs..."
for _ in {1..30}; do
  discover_job_ids
  if [[ "$DISCOVERED_COUNT" -eq "$COUNT" ]]; then
    break
  fi
  sleep 2
done

echo "Monitoring scheduler state transitions with autoscale rebalance..."
iterations=$((MONITOR_SECONDS / POLL_SECONDS))
for ((iter=1; iter<=iterations; iter++)); do
  all_done=1
  now_epoch="$(date +%s)"

  discover_job_ids

  for ((i=1; i<=COUNT; i++)); do
    job_name="${JOB_NAME[$i]}"
    job_id="${JOB_ID[$i]:-}"
    partition="${JOB_PARTITION[$i]}"

    if [[ -n "$job_id" ]]; then
      line="$(squeue -h -j "$job_id" -o "%T|%R" | head -n 1 || true)"
    else
      line=""
    fi

    if [[ -n "$line" ]]; then
      state="${line%%|*}"
      reason="${line#*|}"
      echo -e "${i}\t$(session_for "$i")\t${job_name}\t${partition}\t${state}\t${job_id}\t${reason}" | tee -a "$STATUS_FILE"

      if [[ "$state" == "PENDING" || "$state" == "RUNNING" || "$state" == "CONFIGURING" ]]; then
        all_done=0
      fi

      if [[ "$state" == "PENDING" ]]; then
        age=$((now_epoch - SUBMIT_EPOCH[$i]))
        since_rebalance=$((now_epoch - LAST_REBALANCE_EPOCH[$i]))
        if [[ "$age" -ge "$REBALANCE_AFTER_SECONDS" && "$since_rebalance" -ge "$REBALANCE_COOLDOWN_SECONDS" ]]; then
          target="$(best_partition_for_rebalance "$partition")"
          if [[ -n "$target" && "$target" != "$partition" ]]; then
            echo -e "${i}\t$(session_for "$i")\t${job_name}\t${partition}\tREBALANCE\t${job_id}\tto=${target}" | tee -a "$STATUS_FILE"
            scancel "$job_id" || true
            ATTEMPT[$i]=$((ATTEMPT[$i] + 1))
            LAST_REBALANCE_EPOCH[$i]="$now_epoch"
            launch_index "$i" "$target"
            discover_job_ids
            all_done=0
          fi
        fi
      fi
    else
      echo -e "${i}\t$(session_for "$i")\t${job_name}\t${partition}\tNOT_IN_QUEUE\t-\t-" | tee -a "$STATUS_FILE"
    fi
  done

  if [[ "$all_done" -eq 1 ]]; then
    break
  fi

  sleep "$POLL_SECONDS"
done

echo "Done launching/monitoring batch ${BATCH_ID}."
echo "Attach commands:"
for ((i=1; i<=COUNT; i++)); do
  echo "  tmux attach -t $(session_for "$i")"
done
echo "Inspect statuses: $STATUS_FILE"
