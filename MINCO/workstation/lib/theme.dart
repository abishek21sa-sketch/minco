import 'package:flutter/material.dart';

class MincoTheme {
  static const ink = Color(0xFF17252B);
  static const hospitalBlue = Color(0xFF1B5E78);
  static const paleBlue = Color(0xFFEAF3F6);
  static const paper = Color(0xFFF7F8F6);
  static const panel = Colors.white;
  static const amber = Color(0xFFB5781D);
  static const danger = Color(0xFFB43B38);
  static const success = Color(0xFF3A7652);

  static ThemeData build() {
    final scheme = ColorScheme.fromSeed(
      seedColor: hospitalBlue,
      brightness: Brightness.light,
      surface: paper,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: paper,
      textTheme: const TextTheme(
        headlineSmall: TextStyle(fontWeight: FontWeight.w700, color: ink),
        titleMedium: TextStyle(fontWeight: FontWeight.w700, color: ink),
        bodyMedium: TextStyle(color: ink, height: 1.25),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: panel,
        shape: RoundedRectangleBorder(
          side: const BorderSide(color: Color(0xFFD9E1E3)),
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      dividerColor: const Color(0xFFD9E1E3),
    );
  }
}
