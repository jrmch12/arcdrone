In runpod custom template start command:

```bash

bash -c "echo 'YOUR_PUBLIC_KEY' >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && service ssh start && sleep infinity"

```
