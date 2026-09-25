#!/usr/bin/env bash
# One AWS GPU box for G1 training (Isaac Lab + our MJX pipeline), driven from the Mac.
#
#   scripts/aws_box.sh launch      # create it (once): g6e.2xlarge, 1x L40S 48 GB, Ubuntu 22.04 DLAMI
#   scripts/aws_box.sh status      # state, public IP
#   scripts/aws_box.sh start|stop  # stop whenever idle; the disk (and everything on it) is kept
#   scripts/aws_box.sh ssh [cmd]   # ssh in (the public IP changes on every start)
#   scripts/aws_box.sh sync        # rsync this repo to ~/gtw on the box (no runs/, .venv, third_party)
#   scripts/aws_box.sh train NAME ARGS...   # g1pipe.train in tmux -> ~/gtw/runs/NAME, log ~/NAME.log;
#                                  # the box shuts down 20 min after the run ends
#
# Safety: SSH only, from one IP (security group g1-train-ssh); IMDSv2; the box stops itself
# after 60 min with an idle GPU, no CPU load and nobody logged in. Uses AWS profile sylonik-sso, us-east-1.
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-sylonik-sso} AWS_REGION=${AWS_REGION:-us-east-1}

NAME=${NAME:-g1-train}                       # a second box: NAME=g1-train-2 scripts/aws_box.sh ...
TYPE=${TYPE:-g6e.2xlarge}                   # g6e.xlarge (same L40S, 4 vCPU) when 2xlarge has no capacity
AMI=${AMI:-ami-028e28e7fc9d87d00}            # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260922
SUBNET=${SUBNET:-subnet-0dc6bf369fb078ba3}   # us-east-1a default subnet (g6e is offered in 1a-1d)
SG=sg-01c410eb3570de544                      # g1-train-ssh: tcp/22 from one IP
KEY=g1-train
KEYFILE=~/.ssh/g1-train.pem
DISK_GB=${DISK_GB:-250}
REPO=$(cd "$(dirname "$0")/.." && pwd)

instance_id() {  # the box is found by its Name tag, whatever size or zone it launched in
  aws ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

ip() {
  aws ec2 describe-instances --instance-ids "$(instance_id)" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

USER_DATA=$(cat <<'EOF'
#!/bin/bash
# idle auto-stop: every 5 min, count idle minutes (GPU < 5 %, 5-min load < 0.5, no login); stop at 60
cat >/usr/local/bin/idle-stop.sh <<'EOS'
#!/bin/bash
STATE=/var/tmp/idle-minutes
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
logins=$(who | wc -l)
busy=$(awk '{print ($2 >= 0.5)}' /proc/loadavg)   # installs and CPU work over non-login ssh / tmux
if [ "${util:-0}" -lt 5 ] && [ "$logins" -eq 0 ] && [ "$busy" -eq 0 ]; then n=$(( $(cat $STATE 2>/dev/null || echo 0) + 5 )); else n=0; fi
echo $n > $STATE
if [ "$n" -ge 60 ]; then echo 0 > $STATE; shutdown -h now; fi
EOS
chmod +x /usr/local/bin/idle-stop.sh
echo "*/5 * * * * root /usr/local/bin/idle-stop.sh" > /etc/cron.d/idle-stop
EOF
)

case "${1:-status}" in
  launch)
    if [ -n "$(instance_id)" ]; then echo "already exists: $(instance_id)"; exit 1; fi
    aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
      --subnet-id "$SUBNET" --security-group-ids "$SG" --associate-public-ip-address \
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --instance-initiated-shutdown-behavior stop \
      --user-data "$USER_DATA" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=g1-stairs}]" \
        "ResourceType=volume,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=g1-stairs}]" \
      --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text
    aws ec2 wait instance-running --instance-ids "$(instance_id)"
    echo "running at $(ip)" ;;
  status)
    id=$(instance_id)
    [ -z "$id" ] && { echo "no $NAME instance"; exit 0; }
    aws ec2 describe-instances --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress]' --output text ;;
  start)
    aws ec2 start-instances --instance-ids "$(instance_id)" >/dev/null
    aws ec2 wait instance-running --instance-ids "$(instance_id)"
    echo "running at $(ip)" ;;
  stop)
    aws ec2 stop-instances --instance-ids "$(instance_id)" --query 'StoppingInstances[0].CurrentState.Name' --output text ;;
  ssh)
    shift
    exec ssh -i "$KEYFILE" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=~/.ssh/known_hosts_g1train \
      -o ServerAliveInterval=30 "ubuntu@$(ip)" "$@" ;;
  sync)
    rsync -az --delete -e "ssh -i $KEYFILE -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_g1train" \
      --exclude .venv --exclude .claude --exclude runs --exclude third_party --exclude kaggle_jobs --exclude .git --exclude 'ubuntu@*' \
      --exclude '__pycache__' --exclude results/videos --exclude .env \
      "$REPO/" "ubuntu@$(ip):gtw/" ;;
  train)
    # MuJoCo Warp prints a solver note every step (~14 MB/s); unfiltered it throttles the run and fills the disk
    run=$2; shift 2
    ssh -i "$KEYFILE" -o UserKnownHostsFile=~/.ssh/known_hosts_g1train "ubuntu@$(ip)" \
      "cd gtw && tmux new -d -s $run 'PYTHONPATH=. .venv/bin/python -u -m g1pipe.train --out runs/$run $* 2>&1 \
       | grep --line-buffered -v -e \"iterations limit\" -e \"To disable the print\" -e Warning -e warnings.warn > ~/$run.log; \
       sleep 1200; sudo shutdown -h now'" ;;
  *)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
