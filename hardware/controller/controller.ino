#include <Servo.h>

Servo servo;

int pos;

void setup() {

  servo.attach(D2, 544, 2400);  //D2

  servo.write(30);
  //delay(2000);

  Serial.begin(9600);
}

void loop() {

  if (Serial.available() > 0) {
    int receivedValue = Serial.parseInt();  // Reads the float
    servo.write(receivedValue);
  }
  

  /*for (pos = 0; pos <= 90; pos += 1) { // rotate from 0 degrees to 180 degrees
    // in steps of 1 degree
    servo.write(pos);                   // tell servo to go to position in variable 'pos'
    delay(10);                          // waits 10ms for the servo to reach the position
  }*/

}