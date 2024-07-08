from uvicorn.workers import UvicornWorker


class ProxyUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {"proxy_headers": True, "forwarded_allow_ips": "*"}
