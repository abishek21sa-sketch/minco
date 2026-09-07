import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';
import '../widgets/status_pill.dart';

class MonteCarloRoom extends StatelessWidget {
  const MonteCarloRoom({super.key, required this.simulation});
  final Map<String, dynamic>? simulation;

  @override
  Widget build(BuildContext context) {
    final sim = simulation;
    return Padding(
      padding: const EdgeInsets.all(18),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Row(children: [
          Text('Monte Carlo Room', style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800)),
          SizedBox(width: 12),
          StatusPill('Simulated', level: 'watch'),
          Spacer(),
          Text('Matched stochastic futures', style: TextStyle(fontSize: 11, color: Color(0xFF617178))),
        ]),
        const SizedBox(height: 5),
        const Text('Current and candidate policies are compared on identical random futures to reduce simulation noise.', style: TextStyle(fontSize: 12, color: Color(0xFF617178))),
        const SizedBox(height: 16),
        Expanded(
          child: Row(children: [
            Expanded(
              flex: 7,
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Row(children: [
                      const Text('72-HOUR FUTURE ENVELOPE', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800)),
                      const Spacer(),
                      if (sim != null) Text('${sim['n_common_random_replications']} matched replications', style: const TextStyle(fontSize: 11)),
                    ]),
                    const SizedBox(height: 12),
                    Expanded(child: LayoutBuilder(builder: (context, c) => CustomPaint(size: Size(c.maxWidth, c.maxHeight), painter: _TrajectoryPainter()))),
                  ]),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              flex: 3,
              child: Column(children: [
                Expanded(child: _MetricPanel(title: 'CURRENT POLICY', data: sim?['baseline'] as Map?)),
                const SizedBox(height: 12),
                Expanded(child: _MetricPanel(title: 'CANDIDATE PLAN', data: sim?['candidate'] as Map?, candidate: true)),
              ]),
            ),
          ]),
        ),
      ]),
    );
  }
}

class _MetricPanel extends StatelessWidget {
  const _MetricPanel({required this.title, this.data, this.candidate = false});
  final String title;
  final Map? data;
  final bool candidate;

  @override
  Widget build(BuildContext context) {
    String v(String key, int digits) {
      final n = data?[key];
      return n is num ? n.toDouble().toStringAsFixed(digits) : '—';
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text(title, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w800)),
            const Spacer(),
            StatusPill(candidate ? 'Candidate' : 'Baseline', level: candidate ? 'good' : 'normal'),
          ]),
          const Divider(height: 22),
          _line('Mean ED wait', '${v('mean_ed_wait_hours', 2)} h'),
          _line('P95 ED wait', '${v('p95_ed_wait_hours', 2)} h'),
          _line('Mean boarding', '${v('mean_boarding_hours', 2)} h'),
          _line('P95 boarding', '${v('p95_boarding_hours', 2)} h'),
          _line('Mean max ICU', v('mean_max_icu_occupancy', 1)),
          _line('Mean transfers', v('mean_transfers_out', 1)),
          const Spacer(),
          const Text('SIMULATED RESULT', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w800, color: MincoTheme.amber)),
        ]),
      ),
    );
  }

  Widget _line(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(children: [Expanded(child: Text(label, style: const TextStyle(fontSize: 11))), Text(value, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12))]),
      );
}

class _TrajectoryPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final grid = Paint()..color = const Color(0xFFE4E9EA)..strokeWidth = 1;
    for (var i = 1; i < 6; i++) {
      final y = size.height * i / 6;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }
    for (var i = 1; i < 6; i++) {
      final x = size.width * i / 6;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), grid);
    }
    final capacityY = size.height * .32;
    canvas.drawLine(Offset(0, capacityY), Offset(size.width, capacityY), Paint()..color = MincoTheme.danger..strokeWidth = 2);
    final capText = TextPainter(text: const TextSpan(text: 'STAFFED CAPACITY', style: TextStyle(color: MincoTheme.danger, fontSize: 10, fontWeight: FontWeight.w700)), textDirection: TextDirection.ltr)..layout();
    capText.paint(canvas, Offset(8, capacityY - 18));

    final rng = math.Random(42);
    for (var k = 0; k < 90; k++) {
      final paint = Paint()..color = MincoTheme.hospitalBlue.withValues(alpha: .07)..strokeWidth = 1;
      final path = Path()..moveTo(0, size.height * (.62 + rng.nextDouble() * .12));
      for (var i = 1; i <= 36; i++) {
        final x = size.width * i / 36;
        final trend = -.008 * i;
        final noise = (rng.nextDouble() - .5) * .12;
        final y = size.height * (.68 + trend + noise + .08 * math.sin(i / 4));
        path.lineTo(x, y.clamp(0.0, size.height).toDouble());
      }
      canvas.drawPath(path, paint);
    }
    final candidate = Paint()..color = MincoTheme.success.withValues(alpha: .12)..strokeWidth = 1.2;
    for (var k = 0; k < 45; k++) {
      final path = Path()..moveTo(0, size.height * (.66 + rng.nextDouble() * .08));
      for (var i = 1; i <= 36; i++) {
        final x = size.width * i / 36;
        final y = size.height * (.69 - .003 * i + (rng.nextDouble() - .5) * .08 + .04 * math.sin(i / 5));
        path.lineTo(x, y.clamp(0.0, size.height).toDouble());
      }
      canvas.drawPath(path, candidate);
    }
    final labels = ['NOW', '+12H', '+24H', '+36H', '+48H', '+60H', '+72H'];
    for (var i = 0; i < labels.length; i++) {
      final tp = TextPainter(text: TextSpan(text: labels[i], style: const TextStyle(color: Color(0xFF607177), fontSize: 9)), textDirection: TextDirection.ltr)..layout();
      tp.paint(canvas, Offset(size.width * i / 6 - (i == 0 ? 0 : tp.width / 2), size.height - 15));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
