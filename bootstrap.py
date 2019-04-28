import json
import random
import shlex
import string
import subprocess
import sys
from pathlib import Path


def run(cmd: str):
    subprocess.check_call(cmd, shell=True)


def setup_environment(run_ts: str, ssh_keys=None):
    if 'google.colab' in sys.modules:
        print('Running on Colab')

        # Two steps: global container and Python kernel configuration.

        drive_mount_point = Path('/content/gdrive')
        project_root = drive_mount_point / 'My Drive' / 'rl-attention'
        run_dir = project_root / 'runs' / run_ts

        # Setup ssh access (has its own mechanism to prevent running twice)
        setup_serveo(forward_ports=[6006], ssh_keys=ssh_keys)

        if not drive_mount_point.exists():
            print('Setting up container environment...')

            # Install stable-baselines
            run('apt-get install -qq cmake libopenmpi-dev zlib1g-dev')
            run('pip install -q git+https://github.com/RerRayne/stable-baselines')

            # Run Tensorboard
            run('tensorboard --logdir {} &'.format(shlex.quote(str(run_dir))))

            # Mount Google Drive for logging
            from google.colab import drive
            drive.mount(str(drive_mount_point))
        elif not 'rl-attention' in sys.path:
            print('Setting up Python kernel configuration')
            sys.path.append('rl-attention')
    else:
        print('Running locally')
        project_root = Path('/tmp/rl-attention')
        run_dir = project_root / run_ts

    run_dir.mkdir(exist_ok=True, parents=True)
    return run_dir


def setup_serveo(alias=None, ssh_keys=None, suppress_host_checking=True, forward_ports=None, force=False):
    def get_connection_command():
        with open('serveo.json', 'r') as fp:
            cfg = json.load(fp)

        command = ['ssh']
        if suppress_host_checking:
            command.extend([
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
            ])
        if forward_ports:
            for port in forward_ports:
                command.extend([
                    '-L', '{port}:localhost:{port}'.format(port=port)
                ])
        command.extend([
            '-o', 'ProxyJump=serveo.net',
            'root@' + cfg['alias']
        ])

        return ' '.join(shlex.quote(token) for token in command), cfg['password']

    if not force:
        try:
            cmd, password = get_connection_command()
            print('ssh tunnel is already open, connect via:')
            print(cmd)
            print("Root password: {}".format(password))
            return
        except FileNotFoundError:
            pass

    print('Setting up ssh tunnel...')

    # Generate root password
    password = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(20))

    # Setup sshd
    run('apt-get install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen > /dev/null')
    # Set root password
    run('echo root:{} | chpasswd'.format(password))
    run('mkdir -p /var/run/sshd')
    run('echo "PermitRootLogin yes" >> /etc/ssh/sshd_config')
    run('echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config')
    run('echo "LD_LIBRARY_PATH=/usr/lib64-nvidia" >> /root/.bashrc')
    run('echo "export LD_LIBRARY_PATH" >> /root/.bashrc')

    if ssh_keys is not None:
        # Setup passwordless login
        run('mkdir -p /root/.ssh')
        for ssh_key in ssh_keys:
            run('echo {} >> /root/.ssh/authorized_keys'.format(ssh_key))

    # Run sshd
    run('/usr/sbin/sshd -D &')

    # Create tunnel
    if alias is None:
        alias = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
    run('ssh -o StrictHostKeyChecking=no -R {}:22:localhost:22 serveo.net &'.format(alias))

    with open('serveo.json', 'w') as fp:
        json.dump({
            'alias': alias,
            'password': password,
        }, fp)

    cmd, password = get_connection_command()
    print('ssh tunnel has been set up, connect via:')
    print(cmd)
    print("Root password: {}".format(password))
