import 'dart:async';
import 'dart:convert';

import 'package:grpc/grpc.dart';

class MincoGrpcClient {
  MincoGrpcClient({this.host = '127.0.0.1', this.port = 50551})
      : _channel = ClientChannel(
          host,
          port: port,
          options: const ChannelOptions(credentials: ChannelCredentials.insecure()),
        );

  final String host;
  final int port;
  final ClientChannel _channel;

  static List<int> _encodeVarint(int value) {
    final output = <int>[];
    var remaining = value;
    while (remaining >= 0x80) {
      output.add((remaining & 0x7f) | 0x80);
      remaining >>= 7;
    }
    output.add(remaining);
    return output;
  }

  static ({int value, int next}) _decodeVarint(List<int> bytes, int start) {
    var value = 0;
    var shift = 0;
    var index = start;
    while (index < bytes.length) {
      final byte = bytes[index++];
      value |= (byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return (value: value, next: index);
      shift += 7;
      if (shift > 35) throw const FormatException('Invalid protobuf varint');
    }
    throw const FormatException('Truncated protobuf varint');
  }

  // google.protobuf.StringValue = field #1, wire type 2.
  static List<int> _encodeStringValue(String value) {
    final body = utf8.encode(value);
    return <int>[0x0a, ..._encodeVarint(body.length), ...body];
  }

  static String _decodeStringValue(List<int> bytes) {
    if (bytes.isEmpty) return '';
    if (bytes.first != 0x0a) {
      throw const FormatException('Unexpected protobuf StringValue wire tag');
    }
    final length = _decodeVarint(bytes, 1);
    final end = length.next + length.value;
    if (end > bytes.length) throw const FormatException('Truncated StringValue');
    return utf8.decode(bytes.sublist(length.next, end));
  }

  Future<Map<String, dynamic>> _call(
    String method,
    Map<String, dynamic> payload, {
    Duration timeout = const Duration(seconds: 45),
  }) async {
    final descriptor = ClientMethod<String, String>(
      '/minco.runtime.v1.MincoRuntime/$method',
      (value) => _encodeStringValue(value),
      (bytes) => _decodeStringValue(bytes),
    );
    final call = _channel.createCall(
      descriptor,
      Stream<String>.value(jsonEncode(payload)),
      CallOptions(timeout: timeout),
    );
    final response = await call.response.single;
    final decoded = jsonDecode(response);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('MINCO runtime returned a non-object payload');
    }
    return decoded;
  }

  Future<Map<String, dynamic>> status() => _call('GetStatus', const {});
  Future<Map<String, dynamic>> networkState() => _call('GetNetworkState', const {});
  Future<Map<String, dynamic>> scenarioCatalog() => _call('GetScenarioCatalog', const {});

  Future<Map<String, dynamic>> optimizePlan({
    double riskAlpha = 0.95,
    double riskWeight = 0.40,
    bool usePrimaryJuliaSolver = true,
    String operatingMode = 'normal',
    int networkHospitals = 12,
  }) =>
      _call(
        'OptimizePlan',
        {
          'planning_horizon_periods': 4,
          'n_scenarios': 30,
          'risk_alpha': riskAlpha,
          'risk_weight': riskWeight,
          'risk_posture': 'balanced',
          'use_primary_julia_solver': usePrimaryJuliaSolver,
          'operating_mode': operatingMode,
          'network_hospitals': networkHospitals,
        },
        timeout: const Duration(minutes: 3),
      );

  Future<Map<String, dynamic>> evaluateIntervention({
    required int extraWardBeds,
    required int extraIcuBeds,
    required int extraEdServers,
    required int transferCapacity,
    String operatingMode = 'normal',
  }) =>
      _call(
        'EvaluateIntervention',
        {
          'extra_ward_beds': extraWardBeds,
          'extra_icu_beds': extraIcuBeds,
          'extra_ed_servers': extraEdServers,
          'transfer_out_capacity_per_6h': transferCapacity,
          'n_replications': 120,
          'operating_mode': operatingMode,
        },
        timeout: const Duration(minutes: 3),
      );

  Future<Map<String, dynamic>> reviewDecision(String query) => _call(
        'ReviewDecision',
        {
          'query': query,
          'evidence_scope': ['state', 'plan', 'simulation'],
          'execute_external_llm': false,
        },
      );

  Future<void> close() => _channel.shutdown();
}
