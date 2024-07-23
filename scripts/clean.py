import malevich_coretools as c
import os

try:
    MALEVICH_CORE_USER = os.environ['MALEVICH_CORE_USER']
    MALEVICH_CORE_ACCESS_KEY = os.environ['MALEVICH_CORE_ACCESS_KEY']
    MALEVICH_HOST = os.getenv('MALEVICH_HOST', 'https://nebius.core.malevich.ai/')
except KeyError as e:
    raise KeyError("Environment is configured incorrectly. Missing: " + e.args[0])

c.update_core_credentials(MALEVICH_CORE_USER, MALEVICH_CORE_ACCESS_KEY)
c.set_host_port(MALEVICH_HOST)

c.delete_pipelines()

for i in c.get_run_active_runs().ids:
    print(i)
    try: 
        c.task_stop(i)
    except: 
        pass

