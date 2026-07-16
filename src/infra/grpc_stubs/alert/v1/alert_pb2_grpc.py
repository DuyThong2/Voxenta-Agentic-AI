import grpc

from infra.grpc_stubs.alert.v1 import alert_pb2 as alert__pb2


class AlertServiceStub:
    def __init__(self, channel: grpc.Channel) -> None:
        self.PushAlert = channel.unary_unary(
            "/alert.v1.AlertService/PushAlert",
            request_serializer=alert__pb2.PushAlertRequest.SerializeToString,
            response_deserializer=alert__pb2.PushAlertResponse.FromString,
        )


__all__ = ["AlertServiceStub"]
