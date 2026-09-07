import 'package:flutter/material.dart';
import '../theme.dart';

class StatusPill extends StatelessWidget {
  const StatusPill(this.label, {super.key, this.level = 'normal'});
  final String label;
  final String level;

  @override
  Widget build(BuildContext context) {
    final color = switch (level) {
      'critical' => MincoTheme.danger,
      'watch' => MincoTheme.amber,
      'good' => MincoTheme.success,
      _ => MincoTheme.hospitalBlue,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: .10),
        border: Border.all(color: color.withValues(alpha: .45)),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Text(label.toUpperCase(), style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 11)),
    );
  }
}
