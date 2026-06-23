#include <FastIMU.h>
#include <Wire.h>

// === Bus 0 (D21/D22): dois MPU-9250 =========================================
// Os dois sensores compartilham o mesmo barramento I2C, diferenciados pelo
// endereço definido no pino ADD/SDO de cada um.
// Sensor 1: ADD/SDO -> GND -> 0x68
// Sensor 2: ADD/SDO -> 3V3 -> 0x69
MPU9250 sensor1, sensor2;
calData calib = {0};       // calibração zerada (sem offset); ambos sensores usam a mesma struct
AccelData accel1, accel2;  // guardam a última leitura de aceleração de cada sensor

// === Bus 1 (D32/D33): MPU-6050 ==============================================
// O MPU-6050 fica em um segundo barramento I2C porque seu endereço fixo
// (0x68) colidiria com o Sensor 1 do MPU-9250 caso compartilhasse o Bus 0.
// SDA=D32, SCL=D33, AD0=GND
TwoWire Bus1 = TwoWire(1);
#define MPU6050_ADDR  0x68
#define ACCEL_REG     0x3B    // endereço inicial dos registradores de aceleração (X/Y/Z, 2 bytes cada)
#define PWR_MGMT_1    0x6B    // registrador de gerenciamento de energia
#define ACCEL_SCALE   16384.0f  // LSB/g na escala padrão ±2g do MPU-6050

// A biblioteca FastIMU não suporta o MPU-6050, então ele é lido via
// comunicação I2C manual (sem biblioteca), direto pelos registradores.

// Tira o MPU-6050 do modo sleep (estado padrão ao ligar) escrevendo 0 no
// registrador de gerenciamento de energia.
void mpu6050_init() {
  Bus1.beginTransmission(MPU6050_ADDR);
  Bus1.write(PWR_MGMT_1);
  Bus1.write(0x00);
  Bus1.endTransmission();
}

// Apenas verifica se o sensor responde no barramento (ACK do I2C).
bool mpu6050_test() {
  Bus1.beginTransmission(MPU6050_ADDR);
  return Bus1.endTransmission() == 0;
}

// Lê os 6 bytes de aceleração (X,Y,Z, 2 bytes cada, big-endian) e converte
// para g dividindo pela escala do sensor.
void mpu6050_read(float &ax, float &ay, float &az) {
  Bus1.beginTransmission(MPU6050_ADDR);
  Bus1.write(ACCEL_REG);
  Bus1.endTransmission(false);   // mantém o barramento ativo (sem STOP) para o requestFrom seguinte
  Bus1.requestFrom(MPU6050_ADDR, 6, true);
  ax = (int16_t)((Bus1.read() << 8) | Bus1.read()) / ACCEL_SCALE;
  ay = (int16_t)((Bus1.read() << 8) | Bus1.read()) / ACCEL_SCALE;
  az = (int16_t)((Bus1.read() << 8) | Bus1.read()) / ACCEL_SCALE;
}

void setup() {
  Serial.begin(115200);
  delay(1000);   // tempo para a serial estabilizar antes de imprimir o status

  // Bus 0 - MPU-9250
  Wire.begin(21, 22);
  delay(200);   // tempo de boot do sensor após energizar o barramento
  int err1 = sensor1.init(calib, 0x68);
  Serial.print("Sensor 1 (MPU-9250): ");
  Serial.println(err1 == 0 ? "OK" : "ERRO");

  int err2 = sensor2.init(calib, 0x69);
  Serial.print("Sensor 2 (MPU-9250): ");
  Serial.println(err2 == 0 ? "OK" : "ERRO");

  // Bus 1 - MPU-6050
  Bus1.begin(32, 33);
  delay(200);
  mpu6050_init();
  delay(100);   // tempo para o sensor sair do modo sleep antes do primeiro teste
  Serial.print("Sensor 3 (MPU-6050): ");
  Serial.println(mpu6050_test() ? "OK" : "ERRO");
}

void loop() {
  sensor1.update();
  sensor2.update();
  sensor1.getAccel(&accel1);
  sensor2.getAccel(&accel2);

  float ax3, ay3, az3;
  mpu6050_read(ax3, ay3, az3);

  // formato: x1,y1,z1,x2,y2,z2,x3,y3,z3
  // (sem timestamp: quem recebe via serial marca o instante de chegada)
  Serial.print(accel1.accelX); Serial.print(",");
  Serial.print(accel1.accelY); Serial.print(",");
  Serial.print(accel1.accelZ); Serial.print(",");
  Serial.print(accel2.accelX); Serial.print(",");
  Serial.print(accel2.accelY); Serial.print(",");
  Serial.print(accel2.accelZ); Serial.print(",");
  Serial.print(ax3);           Serial.print(",");
  Serial.print(ay3);           Serial.print(",");
  Serial.println(az3);

  delay(10);   // limita a taxa de envio; a taxa efetiva (~59 Hz) também depende do tempo de leitura dos 3 sensores
}