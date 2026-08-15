"""Stub client cho alert.v1, VIẾT TAY (xem proto/alert/v1/alert.proto).

Không chạy protoc đè lên file này: grpcio-tools hiện hành sinh code gọi
`runtime_version.ValidateProtobufRuntimeVersion`, vốn đòi protobuf >= 5.x, trong khi service ghim
protobuf 4.25.x - file sinh ra sẽ ném lỗi ngay lúc import. Giữ tay cũng khiến mọi thay đổi hợp đồng
hiện rõ trong diff thay vì biến thành một khối bytes.

Tên field phải khớp NGUYÊN VĂN proto của vox-streaming, kể cả kiểu snake_case: tên là thứ code Python
gọi tới, còn số hiệu field mới là thứ đi trên dây. Hai bên từng lệch tên mà trùng số, nên vẫn chạy
đúng - và không ai phát hiện ra suốt thời gian đó.
"""

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, symbol_database

_sym_db = symbol_database.Default()

_file_proto = descriptor_pb2.FileDescriptorProto()
_file_proto.name = "alert/v1/alert.proto"
_file_proto.package = "alert.v1"
_file_proto.syntax = "proto3"

_request = _file_proto.message_type.add()
_request.name = "PushAlertRequest"

for field_name, field_number, field_type in (
    ("session_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("participant_id", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("stream_id", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("alert_type", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("confidence", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
    ("captured_at_ms", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
    ("event_id", 7, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("stream_type", 8, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("detail", 9, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("sequence_no", 10, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
):
    field = _request.field.add()
    field.name = field_name
    field.number = field_number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = field_type

_response = _file_proto.message_type.add()
_response.name = "PushAlertResponse"
_response_field = _response.field.add()
_response_field.name = "received"
_response_field.number = 1
_response_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_response_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL

DESCRIPTOR = descriptor_pool.Default().AddSerializedFile(_file_proto.SerializeToString())

PushAlertRequest = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["PushAlertRequest"])
PushAlertResponse = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["PushAlertResponse"])

_sym_db.RegisterMessage(PushAlertRequest)
_sym_db.RegisterMessage(PushAlertResponse)

__all__ = ["DESCRIPTOR", "PushAlertRequest", "PushAlertResponse"]
