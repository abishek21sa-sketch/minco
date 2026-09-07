class ResourceSnapshot {
  ResourceSnapshot({
    required this.hospitalId,
    required this.resource,
    required this.capacity,
    required this.occupied,
    required this.waiting,
  });

  final String hospitalId;
  final String resource;
  final double capacity;
  final double occupied;
  final double waiting;

  double get utilization => capacity <= 0 ? 0 : occupied / capacity;

  factory ResourceSnapshot.fromJson(Map<String, dynamic> json) => ResourceSnapshot(
        hospitalId: '${json['hospital_id']}',
        resource: '${json['resource']}',
        capacity: (json['capacity'] as num?)?.toDouble() ?? 0,
        occupied: (json['occupied'] as num?)?.toDouble() ?? 0,
        waiting: (json['waiting'] as num?)?.toDouble() ?? 0,
      );
}

class HospitalSnapshot {
  HospitalSnapshot({required this.hospitalId, required this.resources});
  final String hospitalId;
  final List<ResourceSnapshot> resources;

  factory HospitalSnapshot.fromJson(Map<String, dynamic> json) => HospitalSnapshot(
        hospitalId: '${json['hospital_id']}',
        resources: ((json['resources'] as List?) ?? const [])
            .map((e) => ResourceSnapshot.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList(),
      );
}
