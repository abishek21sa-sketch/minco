import 'package:flutter/material.dart';

import '../services/minco_grpc_client.dart';
import '../theme.dart';
import '../widgets/status_pill.dart';

class InterventionComposer extends StatefulWidget {
  const InterventionComposer({
    super.key,
    required this.client,
    required this.plan,
    required this.simulation,
    required this.catalog,
    required this.onPlan,
    required this.onSimulation,
  });
  final MincoGrpcClient client;
  final Map<String, dynamic>? plan;
  final Map<String, dynamic>? simulation;
  final Map<String, dynamic> catalog;
  final ValueChanged<Map<String, dynamic>> onPlan;
  final ValueChanged<Map<String, dynamic>> onSimulation;

  @override
  State<InterventionComposer> createState() => _InterventionComposerState();
}

class _InterventionComposerState extends State<InterventionComposer> {
  double _risk = .40;
  int _ward = 4;
  int _icu = 2;
  int _ed = 1;
  int _transfer = 2;
  String _operatingMode = 'normal';
  int _networkHospitals = 12;
  bool _busy = false;
  String? _error;

  Future<void> _optimize() async {
    setState(() { _busy = true; _error = null; });
    try {
      final plan = await widget.client.optimizePlan(
        riskWeight: _risk,
        usePrimaryJuliaSolver: true,
        operatingMode: _operatingMode,
        networkHospitals: _networkHospitals,
      );
      final sim = await widget.client.evaluateIntervention(
        extraWardBeds: _ward,
        extraIcuBeds: _icu,
        extraEdServers: _ed,
        transferCapacity: _transfer,
        operatingMode: _operatingMode,
      );
      widget.onPlan(plan);
      widget.onSimulation(sim);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(18),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Row(children: [
          Text('Intervention Composer', style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800)),
          SizedBox(width: 12),
          StatusPill('Human review required', level: 'watch'),
          Spacer(),
          Text('Plan → Optimize → Stress Test', style: TextStyle(fontSize: 11, color: Color(0xFF617178))),
        ]),
        const SizedBox(height: 14),
        Expanded(
          child: Row(children: [
            Expanded(flex: 6, child: _timeline()),
            const SizedBox(width: 12),
            Expanded(flex: 4, child: _decisionPanel()),
          ]),
        ),
      ]),
    );
  }

  Widget _timeline() => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('72-HOUR OPERATIONS TIMELINE', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 11)),
            const SizedBox(height: 14),
            const _TimeAxis(),
            const SizedBox(height: 10),
            _modeControls(),
            const SizedBox(height: 10),
            _ActionRow(label: 'ICU', value: '+$_icu beds', start: .18, span: .55, color: MincoTheme.danger),
            _ActionRow(label: 'WARD', value: '+$_ward beds', start: .08, span: .70, color: MincoTheme.hospitalBlue),
            _ActionRow(label: 'ED', value: '+$_ed service server', start: .02, span: .38, color: MincoTheme.success),
            _ActionRow(label: 'TRANSFER', value: '$_transfer / 6h', start: .32, span: .58, color: MincoTheme.amber),
            const SizedBox(height: 18),
            Wrap(spacing: 8, runSpacing: 8, children: [
              _StepButton(label: 'Ward beds', value: _ward, onChanged: (v) => setState(() => _ward = v)),
              _StepButton(label: 'ICU beds', value: _icu, onChanged: (v) => setState(() => _icu = v)),
              _StepButton(label: 'ED servers', value: _ed, onChanged: (v) => setState(() => _ed = v)),
              _StepButton(label: 'Transfers/6h', value: _transfer, onChanged: (v) => setState(() => _transfer = v)),
            ]),
            const Spacer(),
            const Text('Tail-risk posture', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700)),
            Slider(value: _risk, min: 0, max: 1.2, divisions: 12, label: _risk.toStringAsFixed(2), onChanged: (v) => setState(() => _risk = v)),
            Row(children: [
              const Text('Expected-value focus', style: TextStyle(fontSize: 10)),
              const Spacer(),
              Text('CVaR weight ${_risk.toStringAsFixed(2)}', style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w700)),
              const Spacer(),
              const Text('Tail-risk focus', style: TextStyle(fontSize: 10)),
            ]),
          ]),
        ),
      );

  Widget _modeControls() {
    final rawModes = (widget.catalog['operating_modes'] as List?) ?? const [];
    final modes = rawModes
        .whereType<Map>()
        .map((item) => Map<String, dynamic>.from(item))
        .toList();
    if (modes.isEmpty) {
      modes.add({'mode': 'normal', 'display_name': 'Normal operations'});
    }
    final values = modes.map((item) => '${item['mode']}').toSet().toList();
    if (!values.contains(_operatingMode)) values.add(_operatingMode);
    return Row(children: [
      Expanded(
        child: DropdownButtonFormField<String>(
          initialValue: _operatingMode,
          decoration: const InputDecoration(labelText: 'Operating mode', isDense: true),
          items: values.map((mode) {
            Map<String, dynamic>? match;
            for (final item in modes) {
              if (item['mode'] == mode) {
                match = item;
                break;
              }
            }
            return DropdownMenuItem(value: mode, child: Text('${match?['display_name'] ?? mode}'));
          }).toList(),
          onChanged: (value) => setState(() => _operatingMode = value ?? 'normal'),
        ),
      ),
      const SizedBox(width: 10),
      SizedBox(
        width: 150,
        child: DropdownButtonFormField<int>(
          initialValue: _networkHospitals,
          decoration: const InputDecoration(labelText: 'Plan facilities', isDense: true),
          items: const [12, 24, 36].map((value) => DropdownMenuItem(value: value, child: Text('$value hospitals'))).toList(),
          onChanged: (value) => setState(() => _networkHospitals = value ?? 12),
        ),
      ),
    ]);
  }

  Widget _decisionPanel() {
    final plan = widget.plan;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('MINCO OPTIMIZATION', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 11)),
          const SizedBox(height: 8),
          const Text('Julia / JuMP / Gurobi stochastic MILP', style: TextStyle(fontSize: 12)),
          const SizedBox(height: 14),
          if (_error != null) Text(_error!, style: const TextStyle(color: MincoTheme.danger, fontSize: 11)),
          _kv('Solver status', '${plan?['status'] ?? 'NOT RUN'}'),
          _kv('Objective', plan?['objective'] is num ? (plan!['objective'] as num).toStringAsFixed(3) : '—'),
          _kv('Relative gap', plan?['relative_gap'] is num ? (plan!['relative_gap'] as num).toStringAsExponential(2) : '—'),
          _kv('CVaR α', '${plan?['risk_alpha'] ?? '—'}'),
          _kv('Scenarios', '${plan?['n_scenarios'] ?? '—'}'),
          _kv('Operating mode', '${plan?['operating_mode'] ?? _operatingMode}'),
          _kv('Planning network', '${plan?['network_hospitals'] ?? _networkHospitals} facilities'),
          _kv('License posture', '${plan?['solver_license_mode'] ?? 'academic'} research'),
          if (plan?['solver_fallback'] == true)
            Padding(
              padding: const EdgeInsets.only(top: 6, bottom: 4),
              child: Text(
                '${plan?['solver_fallback_reason'] ?? 'Primary solver unavailable; verification oracle used.'}',
                style: const TextStyle(color: MincoTheme.amber, fontSize: 10.5),
              ),
            ),
          const Divider(height: 28),
          const Text('FIRST-STAGE ACTIONS', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 10)),
          const SizedBox(height: 8),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: const Color(0xFFF3F6F5), borderRadius: BorderRadius.circular(6)),
              child: SingleChildScrollView(child: Text(plan == null ? 'Run optimization to populate a feasible surge/staffing plan.' : 'SURGE\n${plan['surge']}\n\nFLEX STAFF BLOCKS\n${plan['flex_blocks']}', style: const TextStyle(fontFamily: 'Consolas', fontSize: 11))),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            height: 44,
            child: FilledButton.icon(
              onPressed: _busy ? null : _optimize,
              icon: _busy ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.calculate_outlined),
              label: Text(_busy ? 'Solving + simulating…' : 'Optimize and stress-test plan'),
            ),
          ),
          const SizedBox(height: 7),
          const Text('No action is executed automatically. This workstation produces decision support for human approval.', style: TextStyle(fontSize: 9.5, color: Color(0xFF69777C))),
        ]),
      ),
    );
  }

  Widget _kv(String k, String v) => Padding(padding: const EdgeInsets.symmetric(vertical: 4), child: Row(children: [Expanded(child: Text(k, style: const TextStyle(fontSize: 11))), Text(v, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700))]));
}

class _TimeAxis extends StatelessWidget {
  const _TimeAxis();
  @override
  Widget build(BuildContext context) => const Row(children: [
        SizedBox(width: 84),
        Expanded(child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [Text('NOW'), Text('+12H'), Text('+24H'), Text('+36H'), Text('+48H'), Text('+60H'), Text('+72H')]))
      ]);
}

class _ActionRow extends StatelessWidget {
  const _ActionRow({required this.label, required this.value, required this.start, required this.span, required this.color});
  final String label;
  final String value;
  final double start;
  final double span;
  final Color color;

  @override
  Widget build(BuildContext context) => SizedBox(
        height: 48,
        child: Row(children: [
          SizedBox(width: 84, child: Text(label, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 11))),
          Expanded(child: LayoutBuilder(builder: (context, c) => Stack(children: [
            Positioned(top: 18, left: 0, right: 0, child: Container(height: 1, color: const Color(0xFFD7DFE1))),
            Positioned(left: c.maxWidth * start, top: 7, width: c.maxWidth * span, child: Container(height: 24, padding: const EdgeInsets.symmetric(horizontal: 8), alignment: Alignment.centerLeft, decoration: BoxDecoration(color: color.withValues(alpha: .12), border: Border.all(color: color.withValues(alpha: .6)), borderRadius: BorderRadius.circular(4)), child: Text(value, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: color)))),
          ]))),
        ]),
      );
}

class _StepButton extends StatelessWidget {
  const _StepButton({required this.label, required this.value, required this.onChanged});
  final String label;
  final int value;
  final ValueChanged<int> onChanged;
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(border: Border.all(color: const Color(0xFFD4DDDF)), borderRadius: BorderRadius.circular(6), color: Colors.white),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Text('$label: $value', style: const TextStyle(fontSize: 10.5, fontWeight: FontWeight.w600)),
          IconButton(visualDensity: VisualDensity.compact, onPressed: value > 0 ? () => onChanged(value - 1) : null, icon: const Icon(Icons.remove, size: 14)),
          IconButton(visualDensity: VisualDensity.compact, onPressed: () => onChanged(value + 1), icon: const Icon(Icons.add, size: 14)),
        ]),
      );
}
