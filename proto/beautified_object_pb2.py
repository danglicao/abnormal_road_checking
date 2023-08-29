# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/beautified_object.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from proto.math import geo_pb2 as proto_dot_math_dot_geo__pb2
from proto.math import color_pb2 as proto_dot_math_dot_color__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1dproto/beautified_object.proto\x12\x05proto\x1a\x14proto/math/geo.proto\x1a\x16proto/math/color.proto\"=\n\x0b\x46ittingFunc\x12\x0e\n\x06xParam\x18\x01 \x03(\x02\x12\x0e\n\x06yParam\x18\x02 \x03(\x02\x12\x0e\n\x06tParam\x18\x03 \x01(\x02\"\x9c\x04\n\x10\x42\x65\x61utifiedObject\x12\n\n\x02id\x18\x01 \x01(\x03\x12\r\n\x05vehId\x18\x02 \x01(\x0c\x12\x0b\n\x03seq\x18\x03 \x01(\x03\x12\x11\n\tkeepFrame\x18\x04 \x01(\x05\x12\x11\n\tis_moving\x18\x05 \x01(\x08\x12&\n\x08position\x18\x06 \x01(\x0b\x32\x14.proto.math.Vector3f\x12#\n\x05shape\x18\x07 \x01(\x0b\x32\x14.proto.math.Vector3f\x12!\n\x04hull\x18\x08 \x01(\x0b\x32\x13.proto.math.Polygon\x12\x13\n\x0borientation\x18\t \x01(\x02\x12\x10\n\x08velocity\x18\n \x01(\x02\x12\x0c\n\x04type\x18\x0b \x01(\x05\x12 \n\x05\x63olor\x18\x0c \x01(\x0b\x32\x11.proto.math.Color\x12\x0f\n\x07heading\x18\r \x01(\x02\x12&\n\nfittingFun\x18\x0e \x01(\x0b\x32\x12.proto.FittingFunc\x12\x10\n\x08timeMeas\x18\x0f \x01(\x03\x12\x13\n\x0bsource_node\x18\x10 \x01(\x0c\x12\x17\n\x0fvehicle_license\x18\x11 \x01(\t\x12\x15\n\rlicense_color\x18\x12 \x01(\x05\x12\x11\n\tobj_color\x18\x13 \x01(\x05\x12!\n\x05plate\x18\x14 \x03(\x0b\x32\x12.proto.LicenePlate\x12\x10\n\x08group_no\x18\x15 \x01(\t\x12\x1b\n\x05glosa\x18\x16 \x01(\x0b\x32\x0c.proto.Glosa\"|\n\x05Glosa\x12\x10\n\x08minSpeed\x18\x01 \x01(\x02\x12\x10\n\x08maxSpeed\x18\x02 \x01(\x02\x12\x17\n\x0fintersection_id\x18\x04 \x01(\t\x12\r\n\x05light\x18\x03 \x01(\t\x12\x15\n\rlike_end_time\x18\x05 \x01(\x05\x12\x10\n\x08veh_area\x18\x06 \x01(\t\"a\n\x0bLicenePlate\x12\r\n\x05\x63olor\x18\x01 \x01(\x05\x12\x18\n\x10\x63olor_confidence\x18\x02 \x01(\x02\x12\x0e\n\x06number\x18\x03 \x01(\t\x12\x19\n\x11number_confidence\x18\x04 \x01(\x02\"a\n\x11\x42\x65\x61utifiedObjects\x12\x10\n\x08time_sub\x18\x01 \x01(\x10\x12\x10\n\x08time_pub\x18\x02 \x01(\x10\x12(\n\x07objects\x18\x03 \x03(\x0b\x32\x17.proto.BeautifiedObjectBY\n\x16\x63om.esurfing.proto.dpeB\x13\x42\x65\x61utifiedObjectDTOP\x00Z(esurfing.com/proto/dpe/beautified_objectb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.beautified_object_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\026com.esurfing.proto.dpeB\023BeautifiedObjectDTOP\000Z(esurfing.com/proto/dpe/beautified_object'
  _FITTINGFUNC._serialized_start=86
  _FITTINGFUNC._serialized_end=147
  _BEAUTIFIEDOBJECT._serialized_start=150
  _BEAUTIFIEDOBJECT._serialized_end=690
  _GLOSA._serialized_start=692
  _GLOSA._serialized_end=816
  _LICENEPLATE._serialized_start=818
  _LICENEPLATE._serialized_end=915
  _BEAUTIFIEDOBJECTS._serialized_start=917
  _BEAUTIFIEDOBJECTS._serialized_end=1014
# @@protoc_insertion_point(module_scope)
