from uvicorn.workers import UvicornWorker


class ProxyUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {"forwarded_allow_ips": "*"}
