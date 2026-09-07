import 'package:flutter/material.dart';

import 'services/minco_grpc_client.dart';
import 'theme.dart';
import 'screens/workstation_shell.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MincoApp());
}

class MincoApp extends StatelessWidget {
  const MincoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MINCO Hospital Operations',
      debugShowCheckedModeBanner: false,
      theme: MincoTheme.build(),
      home: WorkstationShell(client: MincoGrpcClient()),
    );
  }
}
