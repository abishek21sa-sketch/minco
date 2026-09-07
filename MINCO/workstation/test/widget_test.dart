import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:minco_workstation/theme.dart';

void main() {
  test('MINCO theme is light clinical workstation theme', () {
    final theme = MincoTheme.build();
    expect(theme.brightness, equals(Brightness.light));
  });
}
