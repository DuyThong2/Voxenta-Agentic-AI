from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, symbol_database

_sym_db = symbol_database.Default()

_file_proto = descriptor_pb2.FileDescriptorProto()
_file_proto.name = "alert/v1/alert.proto"
_file_proto.package = "alert.v1"
_file_proto.syntax = "proto3"

_request = _file_proto.message_type.add()
_request.name = "PushAlertRequest"

for field_name, field_number, field_type in (
    ("roomId", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("participantId", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("streamId", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("alertType", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("confidence", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
    ("capturedAtMs", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
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
