import 'dart:async';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../services/minco_grpc_client.dart';
import '../theme.dart';
import '../widgets/status_pill.dart';
import 'census_board.dart';
import 'decision_review.dart';
import 'flow_theatre.dart';
import 'intervention_composer.dart';
import 'monte_carlo_room.dart';

class WorkstationShell extends StatefulWidget {
  const WorkstationShell({super.key, required this.client});
  final MincoGrpcClient client;

  @override
  State<WorkstationShell> createState() => _WorkstationShellState();
}

class _WorkstationShellState extends State<WorkstationShell> {
  int _index = 0;
  bool _connecting = true;
  String? _error;
  Map<String, dynamic> _status = const {};
  Map<String, dynamic> _networkState = const {};
  Map<String, dynamic> _scenarioCatalog = const {};
  Map<String, dynamic>? _plan;
  Map<String, dynamic>? _simulation;
  Timer? _clock;
  DateTime _now = DateTime.now();

  static const _labels = ['Census Board', 'Flow Theatre', 'Monte Carlo Room', 'Intervention Composer', 'Decision Review'];

  @override
  void initState() {
    super.initState();
    _refresh();
    _clock = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _now = DateTime.now());
    });
  }

  Future<void> _refresh() async {
    setState(() {
      _connecting = true;
      _error = null;
    });
    try {
      final status = await widget.client.status();
      final state = await widget.client.networkState();
      final catalog = await widget.client.scenarioCatalog();
      if (!mounted) return;
      setState(() {
        _status = status;
        _networkState = state;
        _scenarioCatalog = catalog;
        _connecting = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _connecting = false;
      });
    }
  }

  @override
  void dispose() {
    _clock?.cancel();
    widget.client.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      CensusBoard(state: _networkState, catalog: _scenarioCatalog),
      FlowTheatre(state: _networkState),
      MonteCarloRoom(simulation: _simulation),
      InterventionComposer(
        client: widget.client,
        plan: _plan,
        simulation: _simulation,
        catalog: _scenarioCatalog,
        onPlan: (value) => setState(() => _plan = value),
        onSimulation: (value) => setState(() => _simulation = value),
      ),
      DecisionReview(client: widget.client, plan: _plan, simulation: _simulation),
    ];

    return Scaffold(
      body: Column(
        children: [
          _Header(
            now: _now,
            status: _status,
            connecting: _connecting,
            error: _error,
            onRefresh: _refresh,
          ),
          _SectionTabs(index: _index, onChanged: (value) => setState(() => _index = value)),
          Expanded(
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 180),
              child: KeyedSubtree(key: ValueKey(_index), child: pages[_index]),
            ),
          ),
          Container(
            height: 28,
            padding: const EdgeInsets.symmetric(horizontal: 18),
            decoration: const BoxDecoration(
              color: Color(0xFFEEF2F1),
              border: Border(top: BorderSide(color: Color(0xFFD9E1E3))),
            ),
            child: Row(
              children: [
                Text('Workspace: ${_labels[_index]}', style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
                const Spacer(),
                Text('${_status['data_mode'] ?? 'RUNTIME NOT CONNECTED'}  •  ${_status['event_count'] ?? 0} events', style: const TextStyle(fontSize: 11)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.now, required this.status, required this.connecting, required this.error, required this.onRefresh});
  final DateTime now;
  final Map<String, dynamic> status;
  final bool connecting;
  final String? error;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    final connected = !connecting && error == null && status.isNotEmpty;
    return Container(
      height: 76,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      color: Colors.white,
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(color: MincoTheme.hospitalBlue, borderRadius: BorderRadius.circular(6)),
            child: const Center(child: Text('M', style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.w800))),
          ),
          const SizedBox(width: 12),
          const Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('MINCO', style: TextStyle(fontWeight: FontWeight.w800, letterSpacing: .6, fontSize: 19)),
              Text('Hospital Operations Workstation', style: TextStyle(fontSize: 12, color: Color(0xFF5C6B70))),
            ],
          ),
          const SizedBox(width: 28),
          const VerticalDivider(indent: 17, endIndent: 17),
          const SizedBox(width: 16),
          const Text('MERIDIAN REGIONAL NETWORK', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 12)),
          const SizedBox(width: 12),
          StatusPill(connected ? 'Runtime connected' : connecting ? 'Connecting' : 'Offline', level: connected ? 'good' : 'critical'),
          const SizedBox(width: 8),
          const StatusPill('Academic research', level: 'watch'),
          const Spacer(),
          if (error != null)
            SizedBox(width: 340, child: Text('Runtime unavailable: $error', maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 11, color: MincoTheme.danger))),
          IconButton(onPressed: onRefresh, tooltip: 'Refresh operational state', icon: const Icon(Icons.refresh)),
          const SizedBox(width: 8),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(DateFormat('EEE, MMM d').format(now), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
              Text(DateFormat('HH:mm:ss').format(now), style: const TextStyle(fontSize: 18, fontFeatures: [FontFeature.tabularFigures()])),
            ],
          ),
        ],
      ),
    );
  }
}

class _SectionTabs extends StatelessWidget {
  const _SectionTabs({required this.index, required this.onChanged});
  final int index;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    const items = [
      (Icons.bed_outlined, 'Census Board'),
      (Icons.account_tree_outlined, 'Flow Theatre'),
      (Icons.grain_outlined, 'Monte Carlo Room'),
      (Icons.tune_outlined, 'Intervention Composer'),
      (Icons.fact_check_outlined, 'Decision Review'),
    ];
    return Container(
      height: 48,
      color: const Color(0xFFEAF0F1),
      padding: const EdgeInsets.symmetric(horizontal: 18),
      child: Row(
        children: List.generate(items.length, (i) {
          final active = i == index;
          return InkWell(
            onTap: () => onChanged(i),
            child: Container(
              height: 48,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(border: Border(bottom: BorderSide(color: active ? MincoTheme.hospitalBlue : Colors.transparent, width: 3))),
              child: Row(children: [
                Icon(items[i].$1, size: 18, color: active ? MincoTheme.hospitalBlue : const Color(0xFF64767C)),
                const SizedBox(width: 8),
                Text(items[i].$2, style: TextStyle(fontWeight: active ? FontWeight.w700 : FontWeight.w500, fontSize: 12)),
              ]),
            ),
          );
        }),
      ),
    );
  }
}
