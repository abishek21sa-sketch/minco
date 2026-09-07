import 'package:flutter/material.dart';

import '../theme.dart';
import '../widgets/status_pill.dart';

class FlowTheatre extends StatelessWidget {
  const FlowTheatre({super.key, required this.state});
  final Map<String, dynamic> state;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(18),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Row(children: [
          Text('Patient Flow Theatre', style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800)),
          SizedBox(width: 12),
          StatusPill('Calculated', level: 'good'),
          Spacer(),
          Text('Network topology • queue • capacity', style: TextStyle(fontSize: 11, color: Color(0xFF617178))),
        ]),
        const SizedBox(height: 5),
        const Text('A control-room view of patient movement. Edge labels represent operational flow, not clinical causality.', style: TextStyle(fontSize: 12, color: Color(0xFF617178))),
        const SizedBox(height: 16),
        Expanded(
          child: Card(
            child: LayoutBuilder(builder: (context, constraints) {
              return CustomPaint(
                size: Size(constraints.maxWidth, constraints.maxHeight),
                painter: _FlowPainter(),
                child: const SizedBox.expand(),
              );
            }),
          ),
        ),
      ]),
    );
  }
}

class _FlowPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final line = Paint()..color = const Color(0xFF8FA7AF)..strokeWidth = 2;
    final riskLine = Paint()..color = MincoTheme.amber..strokeWidth = 3;
    final nodes = <String, Offset>{
      'ED ARRIVAL': Offset(size.width * .10, size.height * .50),
      'ED SERVICE': Offset(size.width * .29, size.height * .50),
      'BED REQUEST': Offset(size.width * .48, size.height * .50),
      'WARD': Offset(size.width * .69, size.height * .30),
      'ICU': Offset(size.width * .69, size.height * .70),
      'DISCHARGE': Offset(size.width * .90, size.height * .30),
      'TRANSFER': Offset(size.width * .90, size.height * .70),
    };
    void edge(String a, String b, {bool risk = false}) {
      canvas.drawLine(nodes[a]!, nodes[b]!, risk ? riskLine : line);
    }
    edge('ED ARRIVAL', 'ED SERVICE');
    edge('ED SERVICE', 'BED REQUEST');
    edge('BED REQUEST', 'WARD');
    edge('BED REQUEST', 'ICU', risk: true);
    edge('WARD', 'DISCHARGE');
    edge('ICU', 'TRANSFER', risk: true);

    for (final entry in nodes.entries) {
      final center = entry.value;
      final rect = RRect.fromRectAndRadius(Rect.fromCenter(center: center, width: 124, height: 66), const Radius.circular(8));
      canvas.drawRRect(rect, Paint()..color = Colors.white);
      canvas.drawRRect(rect, Paint()..style = PaintingStyle.stroke..strokeWidth = 1.5..color = const Color(0xFF9CB0B7));
      final text = TextPainter(
        text: TextSpan(text: entry.key, style: const TextStyle(color: MincoTheme.ink, fontWeight: FontWeight.w700, fontSize: 11)),
        textDirection: TextDirection.ltr,
        textAlign: TextAlign.center,
      )..layout(maxWidth: 110);
      text.paint(canvas, center - Offset(text.width / 2, text.height / 2));
    }

    final caption = TextPainter(
      text: const TextSpan(
        text: 'Amber pathway = capacity-sensitive route. Live edge rates are supplied by the runtime in V1.0 integration.',
        style: TextStyle(color: Color(0xFF5F6F74), fontSize: 11),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: size.width - 40);
    caption.paint(canvas, Offset(20, size.height - 34));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
