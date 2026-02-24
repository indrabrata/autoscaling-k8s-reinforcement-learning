# Useful Command

- to sync the local `agent` folder to remote server except some folder, use this command:

```bash
rsync -avz --progress \
                                                                 --exclude='.venv' \
                                                                 --exclude='logs' \
                                                                 --exclude='model' \
                                                                 --exclude='.git' \
                                                                 --exclude='metrics_output' \
                                                                 -e "ssh -p 2222" \
                                                                 ./agent <user>@<ip>:<path-to-agent-folder-on-remote-server>
```

- to get the data from remote server to local machine, use this command:

```bash
 scp -P 2222 -r <user>@<ip>:<path>  /home/ndrz/Projects/autoscaling-k8s-reinforcement-learning/analysis/model/
```

- to run k6 test:

```

# Default: 100 cycles (~50 hours)

k6 run k6-train-unseen-patterns-loop.js

# Custom cycle count

k6 run -e CYCLES=50 k6-train-unseen-patterns-loop.js

# Custom cycle count + custom base URL

k6 run -e CYCLES=50 -e BASE_URL=http://10.34.4.150:30080/api/qfuzzy k6-train-patterns-loop.js

```
