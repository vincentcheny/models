# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: SRC_pro/transfer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='SRC_pro/transfer.proto',
  package='transfer',
  syntax='proto3',
  serialized_options=_b('B\007CYProtoP\001\242\002\003HLW'),
  serialized_pb=_b('\n\x16SRC_pro/transfer.proto\x12\x08transfer\"\x1d\n\rUploadRequest\x12\x0c\n\x04para\x18\x01 \x01(\x0c\"\x1e\n\x0bUploadReply\x12\x0f\n\x07message\x18\x01 \x01(\x0c\"\x1f\n\x0f\x44ownloadRequest\x12\x0c\n\x04para\x18\x01 \x01(\x0c\" \n\rDownloadReply\x12\x0f\n\x07message\x18\x01 \x01(\x0c\x32\x90\x01\n\x08Transfer\x12>\n\nUploadPara\x12\x17.transfer.UploadRequest\x1a\x15.transfer.UploadReply\"\x00\x12\x44\n\x0c\x44ownloadPara\x12\x19.transfer.DownloadRequest\x1a\x17.transfer.DownloadReply\"\x00\x42\x11\x42\x07\x43YProtoP\x01\xa2\x02\x03HLWb\x06proto3')
)




_UPLOADREQUEST = _descriptor.Descriptor(
  name='UploadRequest',
  full_name='transfer.UploadRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='para', full_name='transfer.UploadRequest.para', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=36,
  serialized_end=65,
)


_UPLOADREPLY = _descriptor.Descriptor(
  name='UploadReply',
  full_name='transfer.UploadReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='transfer.UploadReply.message', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=67,
  serialized_end=97,
)


_DOWNLOADREQUEST = _descriptor.Descriptor(
  name='DownloadRequest',
  full_name='transfer.DownloadRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='para', full_name='transfer.DownloadRequest.para', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=99,
  serialized_end=130,
)


_DOWNLOADREPLY = _descriptor.Descriptor(
  name='DownloadReply',
  full_name='transfer.DownloadReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='transfer.DownloadReply.message', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=132,
  serialized_end=164,
)

DESCRIPTOR.message_types_by_name['UploadRequest'] = _UPLOADREQUEST
DESCRIPTOR.message_types_by_name['UploadReply'] = _UPLOADREPLY
DESCRIPTOR.message_types_by_name['DownloadRequest'] = _DOWNLOADREQUEST
DESCRIPTOR.message_types_by_name['DownloadReply'] = _DOWNLOADREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UploadRequest = _reflection.GeneratedProtocolMessageType('UploadRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADREQUEST,
  '__module__' : 'SRC_pro.transfer_pb2'
  # @@protoc_insertion_point(class_scope:transfer.UploadRequest)
  })
_sym_db.RegisterMessage(UploadRequest)

UploadReply = _reflection.GeneratedProtocolMessageType('UploadReply', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADREPLY,
  '__module__' : 'SRC_pro.transfer_pb2'
  # @@protoc_insertion_point(class_scope:transfer.UploadReply)
  })
_sym_db.RegisterMessage(UploadReply)

DownloadRequest = _reflection.GeneratedProtocolMessageType('DownloadRequest', (_message.Message,), {
  'DESCRIPTOR' : _DOWNLOADREQUEST,
  '__module__' : 'SRC_pro.transfer_pb2'
  # @@protoc_insertion_point(class_scope:transfer.DownloadRequest)
  })
_sym_db.RegisterMessage(DownloadRequest)

DownloadReply = _reflection.GeneratedProtocolMessageType('DownloadReply', (_message.Message,), {
  'DESCRIPTOR' : _DOWNLOADREPLY,
  '__module__' : 'SRC_pro.transfer_pb2'
  # @@protoc_insertion_point(class_scope:transfer.DownloadReply)
  })
_sym_db.RegisterMessage(DownloadReply)


DESCRIPTOR._options = None

_TRANSFER = _descriptor.ServiceDescriptor(
  name='Transfer',
  full_name='transfer.Transfer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=167,
  serialized_end=311,
  methods=[
  _descriptor.MethodDescriptor(
    name='UploadPara',
    full_name='transfer.Transfer.UploadPara',
    index=0,
    containing_service=None,
    input_type=_UPLOADREQUEST,
    output_type=_UPLOADREPLY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DownloadPara',
    full_name='transfer.Transfer.DownloadPara',
    index=1,
    containing_service=None,
    input_type=_DOWNLOADREQUEST,
    output_type=_DOWNLOADREPLY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_TRANSFER)

DESCRIPTOR.services_by_name['Transfer'] = _TRANSFER

# @@protoc_insertion_point(module_scope)
