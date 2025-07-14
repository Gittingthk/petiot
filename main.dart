import 'dart:async'; // StreamSubscription을 위해 import
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '스마트 펫 피더',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final DatabaseReference _dbRef = FirebaseDatabase.instance.ref();
  // 데이터 리스너를 관리하기 위한 변수
  late StreamSubscription<DatabaseEvent> _diagLogSubscription;
  late StreamSubscription<DatabaseEvent> _deviceStatusSubscription;


  // 화면에 표시될 상태 변수들
  String _diagnosisResult = "진단 대기 중...";
  String? _imageUrl;
  bool _isOnline = false;
  String _lastSeen = "정보 없음";
  String _currentState = "정보 없음";

  @override
  void initState() {
    super.initState();
    // 위젯이 생성될 때 데이터 리스너를 활성화합니다.
    _activateListeners();
  }

  void _activateListeners() {
    // 1. 진단 로그(diagnosis_log)의 변화를 감지하는 리스너
    _diagLogSubscription = _dbRef.child('diagnosis_log/last_result').onValue.listen((event) {
      final data = event.snapshot.value as Map<dynamic, dynamic>?;
      if (data != null) {
        // 데이터가 변경되면 setState를 호출하여 화면을 다시 그립니다.
        setState(() {
          _diagnosisResult = data['result'] ?? '결과 없음';
          _imageUrl = data['image_url']; // 이미지 URL 업데이트
        });
      }
    });

    // 2. 장치 상태(device_status)의 변화를 감지하는 리스너
    _deviceStatusSubscription = _dbRef.child('device_status').onValue.listen((event) {
      final data = event.snapshot.value as Map<dynamic, dynamic>?;
      if (data != null) {
        setState(() {
          _isOnline = data['is_online'] ?? false;
          _lastSeen = data['last_seen'] ?? '정보 없음';
          _currentState = data['current_state'] ?? '정보 없음';
        });
      }
    });
  }

  // 사료 주기 명령을 Firebase에 보냅니다.
  void _feedPet() {
    // 값을 "NONE"이 아닌, 항상 바뀌는 값(예: 타임스탬프)으로 보내야
    // 장치(라즈베리파이 등)가 변화를 확실히 감지할 수 있습니다.
    final time = DateTime.now().toIso8601String();
    print("🟢 _feedPet() 호출됨, time=$time");
    _dbRef.child('commands/feed').set(DateTime.now().toIso8601String());
    _dbRef.child('commands/feed').set(time)
        .then((_) {
      print("✅ Firebase DB set 성공: commands/feed = $time");
    })
        .catchError((e) {
      print("🚨 Firebase DB set 실패: $e");
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('사료 배급 명령을 보냈습니다.')),
    );
  }

  // 진단 명령을 Firebase에 보냅니다.
  void _diagnosePet() {
    _dbRef.child('commands/diagnose').set(DateTime.now().toIso8601String());
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('백내장 진단 명령을 보냈습니다.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        // 장치 온라인 상태를 제목에 표시
        title: Text('스마트 펫 피더 (${_isOnline ? '온라인' : '오프라인'})'),
        backgroundColor: _isOnline ? Colors.blue : Colors.grey,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            // 버튼들...
            ElevatedButton.icon(
              icon: const Icon(Icons.restaurant),
              label: const Text('지금 사료 주기'),
              onPressed: _feedPet, // onPressed에 직접 함수 연결
              style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 15)),
            ),
            const SizedBox(height: 30),

            ElevatedButton.icon(
              icon: const Icon(Icons.camera_alt),
              label: const Text('사진 찍고 진단하기'),
              onPressed: _diagnosePet, // onPressed에 직접 함수 연결
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                padding: const EdgeInsets.symmetric(vertical: 15),
              ),
            ),
            const SizedBox(height: 40),

            // 진단 결과 표시 영역
            const Text(
              '최근 진단 결과',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            Text(
              _diagnosisResult, // Firebase에서 받아온 결과 표시
              style: const TextStyle(fontSize: 16, color: Colors.black87),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),

            // 촬영된 이미지 표시 영역
            Container(
              height: 200,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Center(
                // _imageUrl이 있으면 네트워크 이미지를, 없으면 텍스트를 보여줌
                child: _imageUrl == null
                    ? const Text('촬영된 이미지가 없습니다.')
                    : Image.network(_imageUrl!, fit: BoxFit.cover,
                  // 이미지 로딩 중/에러 시 처리
                  loadingBuilder: (context, child, progress) {
                    if (progress == null) return child;
                    return const CircularProgressIndicator();
                  },
                  errorBuilder: (context, error, stackTrace) {
                    return const Text('이미지를 불러올 수 없습니다.');
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    // 위젯이 화면에서 사라질 때 리스너를 반드시 해제합니다.
    _diagLogSubscription.cancel();
    _deviceStatusSubscription.cancel();
    super.dispose();
  }
}