#!/usr/bin/env python3
from cereal import car
from math import fabs

from common.conversions import Conversions as CV
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.gm.values import CAR, CruiseButtons, \
                                     CarControllerParams, NO_ASCM
from selfdrive.car.interfaces import CarInterfaceBase

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName
GearShifter = car.CarState.GearShifter

def get_steer_feedforward_erf(angle, speed, ANGLE_COEF, ANGLE_OFFSET, SPEED_OFFSET, SPEED_POWER, SIGMOID_COEF, SPEED_COEF):
  x = ANGLE_COEF * (angle + ANGLE_OFFSET)
  sigmoid = erf(x)
  return (SIGMOID_COEF * sigmoid) / (max(speed - SPEED_OFFSET, 0.1) * SPEED_COEF)**SPEED_POWER

def get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED):
  x = ANGLE * (desired_angle + ANGLE_OFFSET)
  sigmoid = x / (1 + fabs(x))
  return (SIGMOID_SPEED * sigmoid * v_ego) + (SIGMOID * sigmoid) + (SPEED * v_ego)
  
class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    params = CarControllerParams()
    return params.ACCEL_MIN, params.ACCEL_MAX
  
  # Determined by iteratively plotting and minimizing error for f(angle, speed) = steer.
  @staticmethod
  def get_steer_feedforward_volt(desired_angle, v_ego):
    ANGLE = 0.03093722278106523
    ANGLE_OFFSET = 0.#46341000035928637
    SIGMOID_SPEED = 0.07928458395144745
    SIGMOID = 0.4983180128530419
    SPEED = -0.0024896011696167266
    return get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED)

  @staticmethod
  def get_steer_feedforward_acadia(desired_angle, v_ego):
    ANGLE = 0.1314029550298617
    ANGLE_OFFSET = 0.#8317776927522815
    SIGMOID_SPEED = 0.03820691400292691
    SIGMOID = 0.3785405719285944
    SPEED = -0.0010868615264700465
    return get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED)

  @staticmethod
  def get_steer_feedforward_bolt_euv(desired_angle, v_ego):
    ANGLE = 0.0758345580739845
    ANGLE_OFFSET = 0.#31396926577596984
    SIGMOID_SPEED = 0.04367532050459129
    SIGMOID = 0.43144116109994846
    SPEED = -0.002654134623368279
    return get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED)
  
  @staticmethod
  def get_steer_feedforward_bolt(desired_angle, v_ego):
    ANGLE = 0.06370624896135679
    ANGLE_OFFSET = 0.#32536345911579184
    SIGMOID_SPEED = 0.06479105208670367
    SIGMOID = 0.34485246691603205
    SPEED = -0.0010645479469461995
    return get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED)
  
  @staticmethod
  def get_steer_feedforward_silverado(desired_angle, v_ego):
    ANGLE = 0.06539361463056717
    ANGLE_OFFSET = -0.#8390269362439537
    SIGMOID_SPEED = 0.023681877712247515
    SIGMOID = 0.5709779025308087
    SPEED = -0.0016656455765509301
    return get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED)
  
  # Volt determined by iteratively plotting and minimizing error for f(angle, speed) = steer.
  @staticmethod
  def get_steer_feedforward_tahoe(desired_lateral_accel, v_ego):
    ANGLE_COEF = -0.53345154
    ANGLE_OFFSET = 0.
    SPEED_OFFSET = 0.
    SPEED_POWER = 1.
    SIGMOID_COEF = 17.81939495
    SPEED_COEF = -0.47166994
    return get_steer_feedforward_erf(desired_lateral_accel, v_ego, ANGLE_COEF, ANGLE_OFFSET, SPEED_OFFSET, SPEED_POWER, SIGMOID_COEF, SPEED_COEF)
  
  @staticmethod
  def get_steer_feedforward_suburban(desired_angle, v_ego):
    ANGLE = 0.06562376600261893
    ANGLE_OFFSET = 0.#-2.656819831714162
    SIGMOID_SPEED = 0.04648878299738527
    SIGMOID = 0.21826990273744493
    SPEED = -0.001355528078762762
    return get_steer_feedforward_sigmoid(desired_angle, v_ego, ANGLE, ANGLE_OFFSET, SIGMOID_SPEED, SIGMOID, SPEED)

  def get_steer_feedforward_function(self):
    if self.CP.carFingerprint == CAR.VOLT or self.CP.carFingerprint == CAR.VOLT_NR:
      return self.get_steer_feedforward_volt
    elif self.CP.carFingerprint == CAR.ACADIA:
      return self.get_steer_feedforward_acadia
    elif self.CP.carFingerprint == CAR.BOLT_EUV:
      return self.get_steer_feedforward_bolt_euv
    elif self.CP.carFingerprint == CAR.BOLT_NR:
      return self.get_steer_feedforward_bolt
    elif self.CP.carFingerprint == CAR.SILVERADO_NR:
      return self.get_steer_feedforward_silverado
    elif self.CP.carFingerprint == CAR.SUBURBAN:
      return self.get_steer_feedforward_suburban
    elif self.CP.carFingerprint == CAR.TAHOE_NR:
      return self.get_steer_feedforward_tahoe
    else:
      return CarInterfaceBase.get_steer_feedforward_default

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "gm"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.gm)]
    ret.alternativeExperience = 1 # UNSAFE_DISABLE_DISENGAGE_ON_GAS
    ret.pcmCruise = False  # stock cruise control is kept off
    ret.openpilotLongitudinalControl = True # ASCM vehicles use OP for long
    ret.radarOffCan = False # ASCM vehicles (typically) have radar

    # These cars have been put into dashcam only due to both a lack of users and test coverage.
    # These cars likely still work fine. Once a user confirms each car works and a test route is
    # added to selfdrive/car/tests/routes.py, we can remove it from this list.
    ret.dashcamOnly = candidate in {CAR.CADILLAC_ATS, CAR.HOLDEN_ASTRA, CAR.MALIBU, CAR.BUICK_REGAL}

    # Presence of a camera on the object bus is ok.
    # Have to go to read_only if ASCM is online (ACC-enabled cars),
    # or camera is on powertrain bus (LKA cars without ACC).
    
    
    # LKAS only - no radar, no long 
    if candidate in NO_ASCM:
      ret.openpilotLongitudinalControl = False
      ret.radarOffCan = True
    
    # TODO: How Do we detect vehicles using stock cam-based ACC?
      #ret.pcmCruise = True
      
    tire_stiffness_factor = 0.444  # not optimized yet

    # Start with a baseline lateral tuning for all GM vehicles. Override tuning as needed in each model section below.
    ret.minSteerSpeed = 7 * CV.MPH_TO_MS
    ret.lateralTuning.pid.kpBP = [0.]
    ret.lateralTuning.pid.kpV = [0.2]
    ret.lateralTuning.pid.kiBP = [0.]
    ret.lateralTuning.pid.kiV = [0.00]
    ret.lateralTuning.pid.kf = 0.00004   # full torque for 20 deg at 80mph means 0.00007818594
    ret.steerRateCost = 0.5
    ret.steerActuatorDelay = 0.1  # Default delay, not measured yet
    ret.enableGasInterceptor = 0x201 in fingerprint[0]
    # # Check for Electronic Parking Brake
    # TODO: JJS: Add param to cereal
    # ret.hasEPB = 0x230 in fingerprint[0]
    
    # baseline longitudinal tune
    ret.longitudinalTuning.kpBP = [5., 35.]
    ret.longitudinalTuning.kpV = [2.4, 1.5]
    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [0.36]

    ret.steerLimitTimer = 0.4
    ret.radarTimeStep = 0.0667  # GM radar runs at 15Hz instead of standard 20Hz
    
    
    
    if ret.enableGasInterceptor:
      ret.openpilotLongitudinalControl = True

    if candidate == CAR.VOLT or candidate == CAR.VOLT_NR:
      # supports stop and go, but initial engage must be above 18mph (which include conservatism)
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 1607. + STD_CARGO_KG
      ret.wheelbase = 2.69
      ret.steerRatio = 17.7  # Stock 15.7, LiveParameters
      ret.steerRateCost = 1.0
      tire_stiffness_factor = 0.469 # Stock Michelin Energy Saver A/S, LiveParameters
      ret.steerRatioRear = 0.
      ret.centerToFront = 0.45 * ret.wheelbase # from Volt Gen 1

      ret.lateralTuning.pid.kpBP = [0., 40.]
      ret.lateralTuning.pid.kpV = [0., .16]
      ret.lateralTuning.pid.kiBP = [0.]
      ret.lateralTuning.pid.kiV = [.023]
      ret.lateralTuning.pid.kdBP = [0.]
      ret.lateralTuning.pid.kdV = [.6]
      ret.lateralTuning.pid.kf = 1. # !!! ONLY for sigmoid feedforward !!!
      

      # Only tuned to reduce oscillations. TODO.
      ret.longitudinalTuning.kpBP = [5., 15., 35.]
      ret.longitudinalTuning.kpV = [1.25, 1.6, 1.3]
      ret.longitudinalTuning.kiBP = [5., 15., 35.]
      ret.longitudinalTuning.kiV = [0.18, 0.31, 0.34]
      ret.longitudinalTuning.kdBP = [5., 25.]
      ret.longitudinalTuning.kdV = [0.6, 0.0]

    elif candidate == CAR.MALIBU or candidate == CAR.MALIBU_NR:
      # supports stop and go, but initial engage must be above 18mph (which include conservatism)
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 1496. + STD_CARGO_KG
      ret.wheelbase = 2.83
      ret.steerRatio = 15.8
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4  # wild guess

    elif candidate == CAR.HOLDEN_ASTRA:
      ret.mass = 1363. + STD_CARGO_KG
      ret.wheelbase = 2.662
      # Remaining parameters copied from Volt for now
      ret.centerToFront = ret.wheelbase * 0.4
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.steerRatio = 15.7
      ret.steerRatioRear = 0.

    elif candidate == CAR.ACADIA or candidate == CAR.ACADIA_NR:
      ret.minEnableSpeed = -1.  # engage speed is decided by pcm
      ret.mass = 4353. * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.86
      ret.steerRatio = 14.4  # end to end is 13.46
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4
      ret.lateralTuning.pid.kf = 1. # get_steer_feedforward_acadia()

    elif candidate == CAR.BUICK_REGAL:
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 3779. * CV.LB_TO_KG + STD_CARGO_KG  # (3849+3708)/2
      ret.wheelbase = 2.83  # 111.4 inches in meters
      ret.steerRatio = 14.4  # guess for tourx
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4  # guess for tourx

    elif candidate == CAR.CADILLAC_ATS:
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 1601. + STD_CARGO_KG
      ret.wheelbase = 2.78
      ret.steerRatio = 15.3
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.49

    elif candidate == CAR.ESCALADE_ESV:
      ret.minEnableSpeed = -1.  # engage speed is decided by pcm
      ret.mass = 2739. + STD_CARGO_KG
      ret.wheelbase = 3.302
      ret.steerRatio = 17.3
      ret.centerToFront = ret.wheelbase * 0.49
      ret.lateralTuning.pid.kpBP = [10., 41.0]
      ret.lateralTuning.pid.kpV = [0.13, 0.24]
      ret.lateralTuning.pid.kiBP = [10., 41.0]
      ret.lateralTuning.pid.kiV = [0.01, 0.02]
      ret.lateralTuning.pid.kf = 0.000045
      tire_stiffness_factor = 1.0

    elif candidate == CAR.BOLT_NR:
      ret.minEnableSpeed = -1
      ret.minSteerSpeed = 5 * CV.MPH_TO_MS
      ret.mass = 1616. + STD_CARGO_KG
      ret.wheelbase = 2.60096
      ret.steerRatio = 16.8
      ret.steerRatioRear = 0.
      ret.centerToFront = 2.0828 #ret.wheelbase * 0.4 # wild guess
      tire_stiffness_factor = 1.0
      # TODO: Improve stability in turns 
      # still working on improving lateral
      
      # TODO: Should steerRateCost and ActuatorDelay be converted to BPV arrays?
      # TODO: Check if the actuator delay changes based on vehicle speed
      ret.steerRateCost = 0.5
      ret.steerActuatorDelay = 0.
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[10., 41.0], [10., 41.0]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.14, 0.24], [0.01, 0.021]]
      ret.lateralTuning.pid.kdBP = [0.]
      ret.lateralTuning.pid.kdV = [0.5]
      ret.lateralTuning.pid.kf = 1. # for get_steer_feedforward_bolt()
      
      # TODO: Needs refinement for stop and go, doesn't fully stop
      # Assumes the Bolt is using L-Mode for regen braking
      ret.longitudinalTuning.kpBP = [0., 35.]
      ret.longitudinalTuning.kpV = [0.21, 0.46] 
      ret.longitudinalTuning.kiBP = [0., 35.] 
      ret.longitudinalTuning.kiV = [0.22, 0.33]
      ret.stoppingDecelRate = 0.17  # reach stopping target smoothly, brake_travel/s while trying to stop
      ret.stopAccel = 0. # Required acceleraton to keep vehicle stationary
      ret.vEgoStopping = 0.6  # Speed at which the car goes into stopping state, when car starts requesting stopping accel
      ret.vEgoStarting = 0.6  # Speed at which the car goes into starting state, when car starts requesting starting accel,
      # vEgoStarting needs to be > or == vEgoStopping to avoid state transition oscillation
      ret.stoppingControl = True
      ret.longitudinalTuning.deadzoneBP = [0.]
      ret.longitudinalTuning.deadzoneV = [0.]
      
      
    elif candidate == CAR.EQUINOX_NR:
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 3500. * CV.LB_TO_KG + STD_CARGO_KG # (3849+3708)/2
      ret.wheelbase = 2.72 #107.3 inches in meters
      ret.steerRatio = 14.4 # guess for tourx
      ret.steerRatioRear = 0. # unknown online
      ret.centerToFront = ret.wheelbase * 0.4 # wild guess

    elif candidate == CAR.TAHOE_NR:
      ret.minEnableSpeed = -1. # engage speed is decided by pcmFalse
      ret.minSteerSpeed = -1 * CV.MPH_TO_MS
      ret.mass = 5602. * CV.LB_TO_KG + STD_CARGO_KG # (3849+3708)/2
      ret.wheelbase = 2.95 #116 inches in meters
      ret.steerRatio = 16.3 # guess for tourx
      ret.steerRatioRear = 0. # unknown online
      ret.centerToFront = 2.59  # ret.wheelbase * 0.4 # wild guess
      ret.steerActuatorDelay = 0.2
      ret.pcmCruise = True # TODO: see if this resolves cruiseMismatch
      ret.openpilotLongitudinalControl = False # ASCM vehicles use OP for long
      ret.radarOffCan = True # ASCM vehicles (typically) have radar

      # According to JYoung, decrease MAX_LAT_ACCEL if it is understeering
      # friction may need to be increased slowly as well
      # I'm not sure what to do about centering / wandering
      MAX_LAT_ACCEL = 2.5
      ret.lateralTuning.init('torque')
      ret.lateralTuning.torque.useSteeringAngle = True
      ret.lateralTuning.torque.kp = 2.0 / MAX_LAT_ACCEL
      ret.lateralTuning.torque.kf = 1.0 # custom ff
      ret.lateralTuning.torque.kfLeft = 1.0 # custom ff
      ret.lateralTuning.torque.ki = 0.50 / MAX_LAT_ACCEL
      ret.lateralTuning.torque.kd = 6.0 / MAX_LAT_ACCEL
      ret.lateralTuning.torque.friction = 0.01

    elif candidate == CAR.SILVERADO_NR:
      # Thanks skip for the tune!
      ret.minEnableSpeed = -1.
      ret.minSteerSpeed = -1 * CV.MPH_TO_MS
      ret.mass = 2400. + STD_CARGO_KG
      ret.wheelbase = 3.745
      ret.steerRatio = 16.3
      ret.pcmCruise = True # TODO: see if this resolves cruiseMismatch
      ret.centerToFront = ret.wheelbase * .49
      ret.steerRateCost = .4
      ret.steerActuatorDelay = 0.11
      ret.lateralTuning.pid.kpBP = [11., 15.5, 22., 31.0]
      ret.lateralTuning.pid.kpV = [0.12, 0.14, 0.18, 0.20] 
      ret.lateralTuning.pid.kiBP = [0., 22., 26.8, 31.]
      ret.lateralTuning.pid.kiV = [0., 0., 0., 0.035]
      ret.lateralTuning.pid.kdBP = [0.]
      ret.lateralTuning.pid.kdV = [0.005]
      ret.lateralTuning.pid.kf = 0.55 # when turning right. use with get_steer_feedforward_silverado()
      ret.lateralTuning.pid.kfLeft = .4 #  when turning left. use with get_steer_feedforward_silverado()

    elif candidate == CAR.SUBURBAN:
      ret.minEnableSpeed = -1. # engage speed is decided by pcmFalse
      ret.minSteerSpeed = -1 * CV.MPH_TO_MS
      ret.mass = 2731. + STD_CARGO_KG
      ret.wheelbase = 3.302
      ret.steerRatio = 23.2 # LiveParams 17.3 From 2016 spec (unlisted for newer models) TODO: Use LiveParameters to find calculated
      ret.centerToFront = ret.wheelbase * 0.49
      
      ret.pcmCruise = True # TODO: see if this resolves cruiseMismatch
      ret.openpilotLongitudinalControl = False # ASCM vehicles use OP for long
      ret.radarOffCan = True # ASCM vehicles (typically) have radar
      
      ret.steerActuatorDelay = 0.253 # Per Jason Young - I got 0.074
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[10., 41.0], [10., 41.0]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.11, 0.19], [0.02, 0.12]]
      ret.lateralTuning.pid.kpBP = [10., 41.]
      ret.lateralTuning.pid.kpV = [0.11, 0.19]
      ret.lateralTuning.pid.kiBP = [10., 41.]
      ret.lateralTuning.pid.kiV = [0.02, 0.12]
      ret.lateralTuning.pid.kdBP = [0.]
      ret.lateralTuning.pid.kdV = [0.6]
      ret.lateralTuning.pid.kf = 1.0
      ret.steerLimitTimer = 0.5
      # ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[10., 41.0], [10., 41.0]]
      # ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.13, 0.24], [0.01, 0.06]]
      # ret.lateralTuning.pid.kf = 0.000060
      tire_stiffness_factor = 1.0

    elif candidate == CAR.BOLT_EUV:
      ret.minEnableSpeed = -1
      ret.minSteerSpeed = 5 * CV.MPH_TO_MS
      ret.mass = 1616. + STD_CARGO_KG
      ret.wheelbase = 2.60096
      ret.steerRatio = 16.8
      ret.steerRatioRear = 0.
      ret.centerToFront = 2.0828 #ret.wheelbase * 0.4 # wild guess
      tire_stiffness_factor = 1.0
      # TODO: Improve stability in turns 
      # still working on improving lateral
      ret.steerRateCost = 0.5
      ret.steerActuatorDelay = 0.
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[10., 40.0], [0., 40.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.1, 0.22], [0.01, 0.021]]
      ret.lateralTuning.pid.kdBP = [0.]
      ret.lateralTuning.pid.kdV = [0.6]
      ret.lateralTuning.pid.kf = 1. # use with get_feedforward_bolt_euv
      ret.pcmCruise = True # TODO: see if this resolves cruiseMismatch
      ret.openpilotLongitudinalControl = False # Using Stock ACC
      ret.radarOffCan = True # No Radar
      # Note: No Long tuning as we are using stock long
    

         
    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_loopback, self.cp_body)

    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    buttonEvents = []

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons and self.CS.prev_cruise_buttons != CruiseButtons.INIT:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_buttons != CruiseButtons.UNPRESS:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        if not (ret.cruiseState.enabled and ret.standstill):
          be.type = ButtonType.accelCruise  # Suppress resume button if we're resuming from stop so we don't adjust speed.
      elif but == CruiseButtons.DECEL_SET:
        be.type = ButtonType.decelCruise
      elif but == CruiseButtons.CANCEL:
        be.type = ButtonType.cancel
      elif but == CruiseButtons.MAIN:
        be.type = ButtonType.altButton3
      buttonEvents.append(be)

    ret.buttonEvents = buttonEvents
    # TODO: JJS Move this to appropriate place (check other brands)
    EXTRA_GEARS = [GearShifter.sport, GearShifter.low, GearShifter.eco, GearShifter.manumatic]
    events = self.create_common_events(ret, extra_gears = EXTRA_GEARS, pcm_enable=self.CS.CP.pcmCruise)

    if ret.vEgo < self.CP.minEnableSpeed:
      events.add(EventName.belowEngageSpeed)
    if ret.cruiseState.standstill:
      events.add(EventName.resumeRequired)
    if ret.vEgo < self.CP.minSteerSpeed:
      events.add(car.CarEvent.EventName.belowSteerSpeed)

    # handle button presses
    for b in ret.buttonEvents:
      # do enable on both accel and decel buttons
      if b.type in (ButtonType.accelCruise, ButtonType.decelCruise) and not b.pressed:
        events.add(EventName.buttonEnable)
      # do disable on button down
      if b.type == ButtonType.cancel and b.pressed:
        events.add(EventName.buttonCancel)

    ret.events = events.to_msg()

    return ret

  def apply(self, c):
    ret = self.CC.update(c, self.CS)
    return ret
