class UpstreamError(RuntimeError):
    """An upstream data service failed, timed out, or returned something unusable.

    The message is written for the model (and the user) to read, so it names the service and
    says what happened. It is mapped to an MCP tool error at the server boundary.
    """

    def __init__(self, service: str, message: str) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
