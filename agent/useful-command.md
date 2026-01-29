# Useful Command

- To sync the local `agent` folder to remote server except some folder, use this command:

```bash
rsync -avz --progress \
                                                                 --exclude='.venv' \
                                                                 --exclude='logs' \
                                                                 --exclude='model' \
                                                                 --exclude='.git' \
                                                                 -e "ssh -p 2222" \
                                                                 ./agent <user>@<ip>:<path-to-agent-folder-on-remote-server>
```

- to get the data from remote server to local machine, use this command:

```bash
 scp -P 2222 -r <user>@<ip>:<path>  /home/ndrz/Projects/autoscaling-k8s-reinforcement-learning/analysis/model/
```
