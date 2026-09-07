import 'package:flutter/material.dart';

import '../models/workstation_models.dart';
import '../theme.dart';
import '../widgets/status_pill.dart';

class CensusBoard extends StatelessWidget {
  const CensusBoard({super.key, required this.state, this.catalog = const {}});
  final Map<String, dynamic> state;
  final Map<String, dynamic> catalog;

  @override
  Widget build(BuildContext context) {
    final hospitals = ((state['hospitals'] as List?) ?? const [])
        .map((e) => HospitalSnapshot.fromJson(Map<String, dynamic>.from(e as Map)))
        .toList();
    return Padding(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text('Regional Census Board', style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800)),
              const SizedBox(width: 12),
              const StatusPill('Observed / Replay'),
              const SizedBox(width: 8),
              StatusPill('${hospitals.length} facilities', level: 'good'),
              const SizedBox(width: 8),
              StatusPill('${(state['scale_metadata'] as Map?)?['operating_mode'] ?? catalog['active_mode'] ?? 'normal'} mode', level: 'watch'),
              const Spacer(),
              Text('As of ${state['as_of'] ?? '—'}', style: const TextStyle(fontSize: 11, color: Color(0xFF617178))),
            ],
          ),
          const SizedBox(height: 5),
          const Text('Operational state reconstructed from canonical events. Predicted, simulated and optimized quantities are shown elsewhere.', style: TextStyle(fontSize: 12, color: Color(0xFF617178))),
          const SizedBox(height: 16),
          Expanded(
            child: hospitals.isEmpty
                ? const Center(child: Text('Connect the MINCO runtime to load the hospital network.'))
                : GridView.builder(
                    padding: const EdgeInsets.only(right: 4, bottom: 8),
                    gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                      maxCrossAxisExtent: 420,
                      mainAxisExtent: 360,
                      crossAxisSpacing: 12,
                      mainAxisSpacing: 12,
                    ),
                    itemCount: hospitals.length,
                    itemBuilder: (context, index) => _HospitalColumn(hospital: hospitals[index]),
                  ),
          ),
        ],
      ),
    );
  }
}

class _HospitalColumn extends StatelessWidget {
  const _HospitalColumn({required this.hospital});
  final HospitalSnapshot hospital;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Expanded(child: Text(hospital.hospitalId, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16))),
              const StatusPill('Network', level: 'good'),
            ]),
            const SizedBox(height: 4),
            const Text('Adult Operations', style: TextStyle(fontSize: 11, color: Color(0xFF68767B))),
            const SizedBox(height: 14),
            const _ResourceHeader(),
            const Divider(height: 18),
            ...hospital.resources.map((r) => _ResourceRow(resource: r)),
            const Spacer(),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: MincoTheme.paleBlue, borderRadius: BorderRadius.circular(6)),
              child: const Row(children: [
                Icon(Icons.info_outline, size: 16, color: MincoTheme.hospitalBlue),
                SizedBox(width: 7),
                Expanded(child: Text('Capacity values are source/replay state, not forecast capacity.', style: TextStyle(fontSize: 10.5))),
              ]),
            ),
          ],
        ),
      ),
    );
  }
}

class _ResourceHeader extends StatelessWidget {
  const _ResourceHeader();
  @override
  Widget build(BuildContext context) => const Row(children: [
        Expanded(flex: 3, child: Text('RESOURCE', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700))),
        Expanded(child: Text('CAP', textAlign: TextAlign.right, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700))),
        Expanded(child: Text('OCC', textAlign: TextAlign.right, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700))),
        Expanded(child: Text('WAIT', textAlign: TextAlign.right, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700))),
      ]);
}

class _ResourceRow extends StatelessWidget {
  const _ResourceRow({required this.resource});
  final ResourceSnapshot resource;

  @override
  Widget build(BuildContext context) {
    final level = resource.utilization >= .95 ? 'critical' : resource.utilization >= .85 ? 'watch' : 'good';
    final color = level == 'critical' ? MincoTheme.danger : level == 'watch' ? MincoTheme.amber : MincoTheme.success;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Column(children: [
        Row(children: [
          Expanded(flex: 3, child: Row(children: [
            Container(width: 7, height: 7, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
            const SizedBox(width: 7),
            Text(resource.resource, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12)),
          ])),
          Expanded(child: Text(resource.capacity.toStringAsFixed(0), textAlign: TextAlign.right)),
          Expanded(child: Text(resource.occupied.toStringAsFixed(0), textAlign: TextAlign.right)),
          Expanded(child: Text(resource.waiting.toStringAsFixed(0), textAlign: TextAlign.right)),
        ]),
        const SizedBox(height: 5),
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            minHeight: 5,
            value: resource.utilization.clamp(0.0, 1.0).toDouble(),
            backgroundColor: const Color(0xFFE8ECEC),
            color: color,
          ),
        ),
      ]),
    );
  }
}
