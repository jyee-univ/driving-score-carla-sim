# Driving Score CARLA Simulation

## Project Goal

OBD/CAN 기반 운전자 주행 점수화 모델을 CARLA 시뮬레이터로 확장하기 위한 코드 저장소입니다.

본 프로젝트의 목표는 차량 주행 데이터를 이용하여 운전자의 주행 성향을 Safety, Smoothness, Eco 관점에서 정량적으로 평가하는 것입니다.

---

## Overall Flow

1. OBD/CAN 데이터로 차량 내부 조작 신호 기반 점수화 모델 설계
2. cloudpose의 Camera/IMU feature 항목을 참고하여 OBD/CAN에 없는 외부 주행 feature 가상 생성
3. UAH 데이터는 스마트폰 기반 GPS/IMU/영상 주행 상황 feature 참고용으로 활용
4. CARLA는 운전 성향별 시나리오를 생성하고 점수화용 데이터를 추출하는 시뮬레이션 도구로 활용
5. 추출된 CSV에 전처리, Low-pass, FFT, STFT를 적용한 뒤 점수화 함수 실행

---

## Folder Structure

```text
carla_scenarios/
  CARLA 운전 성향별 시나리오 코드

scoring/
  전처리, FFT/STFT, 점수화 함수 코드

data/sample/
  테스트용 샘플 CSV

data/processed/
  전처리 및 점수화 결과 CSV

docs/
  프로젝트 설명 문서

results/
  그래프 및 대시보드 결과