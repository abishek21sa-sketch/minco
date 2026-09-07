import 'package:flutter/material.dart';

import '../services/minco_grpc_client.dart';
import '../theme.dart';
import '../widgets/status_pill.dart';

class DecisionReview extends StatefulWidget {
  const DecisionReview({super.key, required this.client, required this.plan, required this.simulation});
  final MincoGrpcClient client;
  final Map<String, dynamic>? plan;
  final Map<String, dynamic>? simulation;

  @override
  State<DecisionReview> createState() => _DecisionReviewState();
}

class _DecisionReviewState extends State<DecisionReview> {
  final _controller = TextEditingController(text: 'Which assumption is most likely to invalidate this surge plan, and why?');
  Map<String, dynamic>? _result;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _route() async {
    setState(() { _busy = true; _error = null; });
    try {
      final result = await widget.client.reviewDecision(_controller.text);
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final route = _result?['route'] as Map?;
    return Padding(
      padding: const EdgeInsets.all(18),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Row(children: [
          Text('Decision Review Board', style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800)),
          SizedBox(width: 12),
          StatusPill('Claude routed', level: 'normal'),
          Spacer(),
          Text('Evidence-grounded interrogation • no autonomous action', style: TextStyle(fontSize: 11, color: Color(0xFF617178))),
        ]),
        const SizedBox(height: 14),
        Expanded(
          child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Expanded(flex: 4, child: _evidencePanel()),
            const SizedBox(width: 12),
            Expanded(flex: 6, child: Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('ASK THE DECISION, NOT THE MODEL', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800)),
              const SizedBox(height: 8),
              const Text('Routine evidence questions route to the lower-cost Haiku family. Cross-module operational reasoning escalates to Sonnet when the configured budget permits.', style: TextStyle(fontSize: 11, color: Color(0xFF607177))),
              const SizedBox(height: 14),
              TextField(controller: _controller, minLines: 3, maxLines: 5, decoration: const InputDecoration(border: OutlineInputBorder(), labelText: 'Decision challenge / question')),
              const SizedBox(height: 10),
              Row(children: [
                FilledButton.icon(onPressed: _busy ? null : _route, icon: const Icon(Icons.route_outlined), label: Text(_busy ? 'Routing…' : 'Prepare Claude review')),
                const SizedBox(width: 10),
                if (route != null) StatusPill('${route['model_family']} • score ${route['complexity_score']}', level: route['model_family'] == 'sonnet' ? 'watch' : 'good'),
              ]),
              if (_error != null) Padding(padding: const EdgeInsets.only(top: 10), child: Text(_error!, style: const TextStyle(color: MincoTheme.danger, fontSize: 11))),
              const Divider(height: 28),
              Expanded(
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(13),
                  decoration: BoxDecoration(color: const Color(0xFFF3F6F5), borderRadius: BorderRadius.circular(6)),
                  child: SingleChildScrollView(
                    child: Text(
                      _result == null
                          ? 'The deterministic runtime will assemble state + optimization + simulation evidence, select the cost-aware Claude tier, and only then allow the external LLM call. Phase-2 acceptance keeps the paid call disabled by default.'
                          : 'Provider: ${_result!['provider']}\nRoute: ${route?['model_family']}\nReason: ${route?['reason']}\nEvidence bytes: ${_result!['evidence_bytes']}\n\n${_result!['message'] ?? _result!['text'] ?? ''}',
                      style: const TextStyle(fontSize: 11, height: 1.45),
                    ),
                  ),
                ),
              ),
            ])))),
          ]),
        ),
      ]),
    );
  }

  Widget _evidencePanel() => Card(
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('DETERMINISTIC EVIDENCE', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800)),
            const Divider(height: 22),
            _evidenceRow('Operational state', true, 'OBSERVED / REPLAY'),
            _evidenceRow('Stochastic plan', widget.plan != null, 'OPTIMIZED'),
            _evidenceRow('DES comparison', widget.simulation != null, 'SIMULATED'),
            const Spacer(),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: MincoTheme.paleBlue, borderRadius: BorderRadius.circular(6)),
              child: const Text('Claude never computes queueing, Markov, Monte Carlo, or MILP outputs. It interrogates evidence already produced by those engines.', style: TextStyle(fontSize: 10.5)),
            ),
          ]),
        ),
      );

  Widget _evidenceRow(String label, bool available, String provenance) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(children: [
          Icon(available ? Icons.check_circle : Icons.radio_button_unchecked, size: 17, color: available ? MincoTheme.success : const Color(0xFF97A5A9)),
          const SizedBox(width: 8),
          Expanded(child: Text(label, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600))),
          Text(provenance, style: const TextStyle(fontSize: 9, color: Color(0xFF68787E))),
        ]),
      );
}
