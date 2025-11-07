// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//

#include "autoware/multi_object_tracker/tracker/model/tracker_base.hpp"

#include "autoware/multi_object_tracker/object_model/types.hpp"

#include <autoware_utils/geometry/geometry.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

// DEBUG: Add debugging functionality for vehicle tracking analysis
namespace debug_vehicle_tracking
{
// Configuration for debugging - modify these values as needed
// NOTE: Coordinates are EGO-RELATIVE (relative to ego vehicle position)
static constexpr double DEBUG_TARGET_X =
  20.0;  // Target X position relative to ego (meters ahead) - farther vehicle
static constexpr double DEBUG_TARGET_Y =
  0.0;  // Target Y position relative to ego (meters left/right)
static constexpr double DEBUG_AREA_THRESHOLD =
  8.0;                                      // Area threshold in meters (focus on farther vehicle)
static constexpr bool ENABLE_DEBUG = true;  // Set to false to disable debugging

// Log file paths
static const std::string LOG_DIR = "/home/linick/result/251010_debug_update";
static const std::string UPDATE_LOG_FILE = LOG_DIR + "/vehicle_update_analysis.log";
static const std::string DETAILED_LOG_FILE = LOG_DIR + "/vehicle_detailed_debug.log";

// Function to ensure directory exists
void ensureLogDirectory()
{
  if (!debug_vehicle_tracking::ENABLE_DEBUG) return;
  std::filesystem::create_directories(debug_vehicle_tracking::LOG_DIR);
}

// Function to get current timestamp
std::string getCurrentTimestamp()
{
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  ss << "." << std::setfill('0') << std::setw(3) << ms.count();
  return ss.str();
}

// Function to write to log file
void writeToLog(const std::string & filename, const std::string & message)
{
  if (!debug_vehicle_tracking::ENABLE_DEBUG) return;
  ensureLogDirectory();

  std::ofstream log_file(filename, std::ios::app);
  if (log_file.is_open()) {
    log_file << "[" << getCurrentTimestamp() << "] " << message << std::endl;
    log_file.close();
    // DEBUG: Also print to stderr for immediate diagnosis
    std::cerr << "DEBUG_LOG: " << message.substr(0, 100) << std::endl;
  } else {
    // Debug: write to stderr if file can't be opened
    std::cerr << "DEBUG: Failed to open log file: " << filename << std::endl;
    std::cerr << "DEBUG: Message was: " << message.substr(0, 100) << "..." << std::endl;
  }
}

// Function to check if tracker is a vehicle (using the same logic as association.cpp)
bool isVehicleTracker(const autoware::multi_object_tracker::Tracker & tracker)
{
  using Label = autoware_perception_msgs::msg::ObjectClassification;
  const auto tracker_label = tracker.getHighestProbLabel();
  const bool is_vehicle_tracker = tracker_label == Label::CAR || tracker_label == Label::BUS ||
                                  tracker_label == Label::TRUCK || tracker_label == Label::TRAILER;
  return is_vehicle_tracker;
}

// Function to transform object position to ego-relative coordinates
std::pair<double, double> transformToEgoRelative(
  const autoware::multi_object_tracker::types::DynamicObject & object,
  const geometry_msgs::msg::Pose & ego_pose)
{
  // Get object position in global frame
  const double obj_x = object.pose.position.x;
  const double obj_y = object.pose.position.y;

  // Get ego position and orientation in global frame
  const double ego_x = ego_pose.position.x;
  const double ego_y = ego_pose.position.y;
  const double ego_yaw = tf2::getYaw(ego_pose.orientation);

  // Transform object position to ego vehicle's local coordinate frame
  // Step 1: Translate to ego-centered coordinates
  const double dx_global = obj_x - ego_x;
  const double dy_global = obj_y - ego_y;

  // Step 2: Rotate to ego vehicle's orientation (inverse rotation)
  const double cos_yaw = std::cos(-ego_yaw);  // Negative for inverse rotation
  const double sin_yaw = std::sin(-ego_yaw);

  const double relative_x =
    dx_global * cos_yaw - dy_global * sin_yaw;  // Forward/backward relative to ego
  const double relative_y =
    dx_global * sin_yaw + dy_global * cos_yaw;  // Left/right relative to ego

  return std::make_pair(relative_x, relative_y);
}

// Function to check if tracker is in debug area (ego-relative coordinates)
bool isTrackerInDebugArea(
  const autoware::multi_object_tracker::Tracker & tracker,
  const autoware::multi_object_tracker::types::DynamicObject & tracker_object,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose)
{
  if (!debug_vehicle_tracking::ENABLE_DEBUG) return false;

  // Only debug vehicle trackers
  if (!isVehicleTracker(tracker)) return false;

  // If no ego pose available, fall back to global coordinate check
  if (!ego_pose) {
    const double dx = tracker_object.pose.position.x - debug_vehicle_tracking::DEBUG_TARGET_X;
    const double dy = tracker_object.pose.position.y - debug_vehicle_tracking::DEBUG_TARGET_Y;
    const double distance = std::sqrt(dx * dx + dy * dy);
    return distance <= debug_vehicle_tracking::DEBUG_AREA_THRESHOLD;
  }

  // Transform to ego-relative coordinates
  auto [relative_x, relative_y] = transformToEgoRelative(tracker_object, *ego_pose);

  // Check if object is at target relative position
  const double dx = relative_x - debug_vehicle_tracking::DEBUG_TARGET_X;
  const double dy = relative_y - debug_vehicle_tracking::DEBUG_TARGET_Y;
  const double distance = std::sqrt(dx * dx + dy * dy);
  return distance <= debug_vehicle_tracking::DEBUG_AREA_THRESHOLD;
}

// Function to convert UUID to string
std::string uuidToString(const unique_identifier_msgs::msg::UUID & uuid)
{
  std::stringstream ss;
  for (size_t i = 0; i < uuid.uuid.size(); ++i) {
    ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(uuid.uuid[i]);
    if (i == 3 || i == 5 || i == 7 || i == 9) ss << "-";
  }
  return ss.str();
}

// Function to format object state with ego-relative coordinates only
std::string formatObjectState(
  const std::string & prefix, const autoware::multi_object_tracker::types::DynamicObject & object,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose)
{
  std::stringstream ss;
  ss << prefix;

  // Ego-relative coordinates if ego pose available
  if (ego_pose) {
    auto [rel_x, rel_y] = transformToEgoRelative(object, *ego_pose);
    ss << " - Ego-rel: (" << std::fixed << std::setprecision(3) << rel_x << ", " << rel_y << ")";
  } else {
    ss << " - No ego pose available";
  }

  // Velocity
  ss << ", Vel: (" << (object.kinematics.has_twist ? object.twist.linear.x : 0.0) << ", "
     << (object.kinematics.has_twist ? object.twist.linear.y : 0.0) << ")";

  // Dimensions
  ss << ", Dims: (" << object.shape.dimensions.x << ", " << object.shape.dimensions.y << ", "
     << object.shape.dimensions.z << ")";

  return ss.str();
}

// Function to log complete update summary
void logUpdateSummary(
  const std::string & uuid, const std::string & update_method,
  const autoware::multi_object_tracker::types::DynamicObject & before_state,
  const autoware::multi_object_tracker::types::DynamicObject & after_state,
  const autoware::multi_object_tracker::types::DynamicObject & measurement,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose,
  const std::string & additional_details = "")
{
  if (!debug_vehicle_tracking::ENABLE_DEBUG) return;

  std::stringstream summary;
  summary << "=== VEHICLE UPDATE SUMMARY ===\n";
  summary << "UUID: " << uuid << " | METHOD: " << update_method;
  if (!additional_details.empty()) {
    summary << " | DETAILS: " << additional_details;
  }
  summary << "\n";

  // Add ego-relative coordinates
  if (ego_pose) {
    auto [before_rel_x, before_rel_y] = transformToEgoRelative(before_state, *ego_pose);
    auto [after_rel_x, after_rel_y] = transformToEgoRelative(after_state, *ego_pose);
    auto [meas_rel_x, meas_rel_y] = transformToEgoRelative(measurement, *ego_pose);

    summary << "BEFORE - EgoRel: (" << std::fixed << std::setprecision(3) << before_rel_x << ", "
            << before_rel_y << ")"
            << ", Vel: (" << (before_state.kinematics.has_twist ? before_state.twist.linear.x : 0.0)
            << ", " << (before_state.kinematics.has_twist ? before_state.twist.linear.y : 0.0)
            << ")"
            << ", Dims: (" << before_state.shape.dimensions.x << ", "
            << before_state.shape.dimensions.y << ", " << before_state.shape.dimensions.z << ")\n";
    summary << "MEASUREMENT - EgoRel: (" << meas_rel_x << ", " << meas_rel_y << ")"
            << ", Vel: (" << (measurement.kinematics.has_twist ? measurement.twist.linear.x : 0.0)
            << ", " << (measurement.kinematics.has_twist ? measurement.twist.linear.y : 0.0) << ")"
            << ", Dims: (" << measurement.shape.dimensions.x << ", "
            << measurement.shape.dimensions.y << ", " << measurement.shape.dimensions.z << ")\n";
    summary << "AFTER - EgoRel: (" << after_rel_x << ", " << after_rel_y << ")"
            << ", Vel: (" << (after_state.kinematics.has_twist ? after_state.twist.linear.x : 0.0)
            << ", " << (after_state.kinematics.has_twist ? after_state.twist.linear.y : 0.0) << ")"
            << ", Dims: (" << after_state.shape.dimensions.x << ", "
            << after_state.shape.dimensions.y << ", " << after_state.shape.dimensions.z << ")\n";
    summary << "CHANGE - ΔEgoRel: (" << (after_rel_x - before_rel_x) << ", "
            << (after_rel_y - before_rel_y) << ")"
            << ", ΔVel: ("
            << ((after_state.kinematics.has_twist ? after_state.twist.linear.x : 0.0) -
                (before_state.kinematics.has_twist ? before_state.twist.linear.x : 0.0))
            << ", "
            << ((after_state.kinematics.has_twist ? after_state.twist.linear.y : 0.0) -
                (before_state.kinematics.has_twist ? before_state.twist.linear.y : 0.0))
            << ")"
            << ", ΔDims: (" << (after_state.shape.dimensions.x - before_state.shape.dimensions.x)
            << ", " << (after_state.shape.dimensions.y - before_state.shape.dimensions.y) << ", "
            << (after_state.shape.dimensions.z - before_state.shape.dimensions.z) << ")\n";
  } else {
    // Fallback to global coordinates if no ego pose
    summary << "BEFORE - Pos: (" << std::fixed << std::setprecision(3)
            << before_state.pose.position.x << ", " << before_state.pose.position.y << ", "
            << before_state.pose.position.z << ")"
            << ", Vel: (" << (before_state.kinematics.has_twist ? before_state.twist.linear.x : 0.0)
            << ", " << (before_state.kinematics.has_twist ? before_state.twist.linear.y : 0.0)
            << ")"
            << ", Dims: (" << before_state.shape.dimensions.x << ", "
            << before_state.shape.dimensions.y << ", " << before_state.shape.dimensions.z << ")\n";
    summary << "MEASUREMENT - Pos: (" << measurement.pose.position.x << ", "
            << measurement.pose.position.y << ", " << measurement.pose.position.z << ")"
            << ", Vel: (" << (measurement.kinematics.has_twist ? measurement.twist.linear.x : 0.0)
            << ", " << (measurement.kinematics.has_twist ? measurement.twist.linear.y : 0.0) << ")"
            << ", Dims: (" << measurement.shape.dimensions.x << ", "
            << measurement.shape.dimensions.y << ", " << measurement.shape.dimensions.z << ")\n";
    summary << "AFTER - Pos: (" << after_state.pose.position.x << ", "
            << after_state.pose.position.y << ", " << after_state.pose.position.z << ")"
            << ", Vel: (" << (after_state.kinematics.has_twist ? after_state.twist.linear.x : 0.0)
            << ", " << (after_state.kinematics.has_twist ? after_state.twist.linear.y : 0.0) << ")"
            << ", Dims: (" << after_state.shape.dimensions.x << ", "
            << after_state.shape.dimensions.y << ", " << after_state.shape.dimensions.z << ")\n";
    summary << "CHANGE - ΔPos: (" << (after_state.pose.position.x - before_state.pose.position.x)
            << ", " << (after_state.pose.position.y - before_state.pose.position.y) << ")"
            << ", ΔVel: ("
            << ((after_state.kinematics.has_twist ? after_state.twist.linear.x : 0.0) -
                (before_state.kinematics.has_twist ? before_state.twist.linear.x : 0.0))
            << ", "
            << ((after_state.kinematics.has_twist ? after_state.twist.linear.y : 0.0) -
                (before_state.kinematics.has_twist ? before_state.twist.linear.y : 0.0))
            << ")"
            << ", ΔDims: (" << (after_state.shape.dimensions.x - before_state.shape.dimensions.x)
            << ", " << (after_state.shape.dimensions.y - before_state.shape.dimensions.y) << ", "
            << (after_state.shape.dimensions.z - before_state.shape.dimensions.z) << ")\n";
  }
  summary << "==============================";

  writeToLog(UPDATE_LOG_FILE, summary.str());
}

}  // namespace debug_vehicle_tracking

namespace
{
float updateProbability(
  const float & prior, const float & true_positive, const float & false_positive,
  const bool clamp = true)
{
  float probability =
    (prior * true_positive) / (prior * true_positive + (1 - prior) * false_positive);

  if (clamp) {
    // Normalize the probability to [0.1, 0.999]
    constexpr float max_updated_probability = 0.999;
    constexpr float min_updated_probability = 0.100;
    probability = std::clamp(probability, min_updated_probability, max_updated_probability);
  }

  return probability;
}
float decayProbability(const float & prior, const float & delta_time)
{
  constexpr float minimum_probability = 0.001;
  const float decay_rate = log(0.5f) / 0.5f;  // half-life (50% decay) of 0.5s
  return std::max(prior * std::exp(decay_rate * delta_time), minimum_probability);
}
}  // namespace

namespace autoware::multi_object_tracker
{

Tracker::Tracker(const rclcpp::Time & time, const types::DynamicObject & detected_object)
: no_measurement_count_(0),
  total_no_measurement_count_(0),
  total_measurement_count_(1),
  last_update_with_measurement_time_(time),
  object_(detected_object)
{
  // Generate random number
  std::mt19937 gen(std::random_device{}());
  std::independent_bits_engine<std::mt19937, 8, uint8_t> bit_eng(gen);
  unique_identifier_msgs::msg::UUID uuid_msg;
  std::generate(uuid_msg.uuid.begin(), uuid_msg.uuid.end(), bit_eng);
  object_.uuid = uuid_msg;

  // Initialize existence probabilities
  existence_probabilities_.resize(types::max_channel_size, 0.001);
  total_existence_probability_ = 0.001;
  classification_ = detected_object.classification;
}

void Tracker::initializeExistenceProbabilities(
  const uint & channel_index, const float & existence_probability)
{
  // The initial existence probability is normalized to [0.1, 0.999]
  // to avoid the existence probability being too low or too high
  // and to avoid the existence probability being too close to 0 or 1
  constexpr float max_probability = 0.999;
  constexpr float min_probability = 0.100;
  const float clamped_existence_probability =
    std::clamp(existence_probability, min_probability, max_probability);

  // existence probability on each channel
  existence_probabilities_[channel_index] = clamped_existence_probability;

  // total existence probability
  total_existence_probability_ = clamped_existence_probability;
}

void Tracker::updateTotalExistenceProbability(const float & existence_probability)
{
  total_existence_probability_ =
    updateProbability(total_existence_probability_, existence_probability, 0.2);
}

void Tracker::mergeExistenceProbabilities(std::vector<float> existence_probabilities)
{
  // existence probability on each channel
  for (size_t i = 0; i < existence_probabilities.size(); ++i) {
    // take larger value
    existence_probabilities_[i] = std::max(existence_probabilities_[i], existence_probabilities[i]);
  }
}

bool Tracker::updateWithMeasurement(
  const types::DynamicObject & object, const rclcpp::Time & measurement_time,
  const types::InputChannel & channel_info, bool significant_shape_change,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose)
{
  // DEBUG: Always log that updateWithMeasurement was called (for diagnosis)
  static bool first_call = true;
  if (first_call && debug_vehicle_tracking::ENABLE_DEBUG) {
    first_call = false;
    std::stringstream init_msg;
    init_msg << "DEBUG SYSTEM INITIALIZED: updateWithMeasurement called for first time";
    debug_vehicle_tracking::writeToLog(debug_vehicle_tracking::UPDATE_LOG_FILE, init_msg.str());
    std::cout << "DEBUG: First updateWithMeasurement call - log system active" << std::endl;
  }

  // DEBUG: Check if this tracker is in our debug area (check tracker state, not measurement)
  const bool is_debug_object =
    debug_vehicle_tracking::isTrackerInDebugArea(*this, object_, ego_pose);

  // Store tracker state before update for debug comparison
  types::DynamicObject tracker_before_state;
  std::string update_method;
  std::string update_details;

  if (is_debug_object) {
    tracker_before_state = object_;  // Store tracker's current state
  }

  // Update existence probability
  {
    no_measurement_count_ = 0;
    ++total_measurement_count_;

    // existence probability on each channel
    const float delta_time =
      std::abs((measurement_time - last_update_with_measurement_time_).seconds());
    constexpr float probability_true_detection = 0.9;
    constexpr float probability_false_detection = 0.2;

    // update measured channel probability without decay
    const uint & channel_index = channel_info.index;
    existence_probabilities_[channel_index] = updateProbability(
      existence_probabilities_[channel_index], probability_true_detection,
      probability_false_detection);

    // decay other channel probabilities
    for (size_t i = 0; i < existence_probabilities_.size(); ++i) {
      if (i != channel_index) {
        existence_probabilities_[i] = decayProbability(existence_probabilities_[i], delta_time);
      }
    }

    // update total existence probability
    total_existence_probability_ = updateProbability(
      total_existence_probability_, object.existence_probability * probability_true_detection,
      probability_false_detection);
  }

  last_update_with_measurement_time_ = measurement_time;

  // Update classification
  if (
    channel_info.trust_classification &&
    autoware::object_recognition_utils::getHighestProbLabel(object.classification) !=
      autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) {
    updateClassification(object.classification);
  }

  // Update orientation availability
  if (object.kinematics.orientation_availability == types::OrientationAvailability::AVAILABLE) {
    // if the incoming object is AVAILABLE, set the orientation availability to AVAILABLE
    object_.kinematics.orientation_availability = types::OrientationAvailability::AVAILABLE;
  } else if (
    object.kinematics.orientation_availability == types::OrientationAvailability::SIGN_UNKNOWN &&
    object_.kinematics.orientation_availability == types::OrientationAvailability::UNAVAILABLE) {
    // if the incoming object is SIGN_UNKNOWN and the tracker is UNAVAILABLE, set the orientation
    // availability to SIGN_UNKNOWN
    object_.kinematics.orientation_availability = types::OrientationAvailability::SIGN_UNKNOWN;
  }

  // Main update logic - determine which path to take
  if (!significant_shape_change) {
    if (is_debug_object) {
      update_method = "NORMAL_UPDATE";
      update_details = "No significant shape change detected";
    }

    // Input normal measurement for EMA
    ema_shape_.processNormalMeasurement(object);

    // Store state before measure for debug logging
    types::DynamicObject before_measure_state;
    if (is_debug_object) {
      before_measure_state = getTrackedObjectDebug();
    }

    // Update object normally
    measure(object, measurement_time, channel_info);
    object_.trust_extension = object.trust_extension;

    // For normal measure, log using compact format
    if (is_debug_object) {
      types::DynamicObject after_measure_state;
      after_measure_state = getTrackedObjectDebug();

      debug_vehicle_tracking::logUpdateSummary(
        debug_vehicle_tracking::uuidToString(object_.uuid), update_method, before_measure_state,
        after_measure_state, object, ego_pose, update_details);
      return true;  // Skip the general logging since we did detailed logging here
    }
  } else {
    ema_shape_.processNoisyMeasurement(object);
    if (ema_shape_.isStable()) {
      const auto & smoothed_shape_const = ema_shape_.getShape();
      autoware_perception_msgs::msg::Shape smoothed_shape(smoothed_shape_const);
      if (is_debug_object) {
        update_method = "EMA_STABLE_UPDATE";
        std::stringstream ss;
        ss << "EMA stable, smoothed shape dims(" << std::fixed << std::setprecision(3)
           << smoothed_shape.dimensions.x << ", " << smoothed_shape.dimensions.y << ", "
           << smoothed_shape.dimensions.z << ")";
        update_details = ss.str();
      }

      setObjectShape(smoothed_shape);
      // Update object normally
      auto smoothed_object = object;
      smoothed_object.shape = smoothed_shape;
      measure(smoothed_object, measurement_time, channel_info);
      object_.trust_extension = smoothed_object.trust_extension;

      // Renew ema_shape_
      ema_shape_.clear();
    } else {
      const auto current_shape = object_.shape;

      if (is_debug_object) {
        update_method = "CONDITIONED_UPDATE";
        std::stringstream ss;
        ss << "EMA not stable, using conditioned update with current dims(" << std::fixed
           << std::setprecision(3) << current_shape.dimensions.x << ", "
           << current_shape.dimensions.y << ", " << current_shape.dimensions.z << ")";
        update_details = ss.str();
      }

      // Get predicted object
      types::DynamicObject predicted_object;
      getTrackedObject(measurement_time, predicted_object);

      // Perform conditioned update and capture strategy
      std::string update_strategy = "NORMAL_UPDATE";
      conditionedUpdate(
        object, predicted_object, current_shape, measurement_time, channel_info, update_strategy,
        is_debug_object);

      if (is_debug_object) {
        update_method = "CONDITIONED_UPDATE";
        update_details = "Strategy: " + update_strategy;
      }
    }
  }

  // Update object status
  getTrackedObject(measurement_time, object_);

  // update time
  object_.time = measurement_time;

  // Log comprehensive update summary for non-normal updates only
  // (Normal updates already logged detailed info above)
  if (is_debug_object && update_method != "NORMAL_UPDATE") {
    debug_vehicle_tracking::logUpdateSummary(
      debug_vehicle_tracking::uuidToString(object_.uuid), update_method, tracker_before_state,
      object_, object, ego_pose, update_details);
  }

  return true;
}

bool Tracker::updateWithoutMeasurement(const rclcpp::Time & timestamp)
{
  // Update existence probability
  ++no_measurement_count_;
  ++total_no_measurement_count_;
  {
    // decay existence probability
    float const delta_time = (timestamp - last_update_with_measurement_time_).seconds();
    for (float & existence_probability : existence_probabilities_) {
      existence_probability = decayProbability(existence_probability, delta_time);
    }
    total_existence_probability_ = decayProbability(total_existence_probability_, delta_time);
  }

  // Update object status
  getTrackedObject(timestamp, object_);

  return true;
}

bool Tracker::createPseudoMeasurement(
  const types::DynamicObject & meas, types::DynamicObject & pred,
  const autoware_perception_msgs::msg::Shape & smoothed_shape, const bool enlarge_covariance)
{
  // Apply linear fall‑off weight on dist square
  const double dx = meas.pose.position.x - pred.pose.position.x;
  const double dy = meas.pose.position.y - pred.pose.position.y;
  const double dist2 = dx * dx + dy * dy;
  constexpr double d_max_square_inv = 1 / 2.0;  // saturate when distance overs 1.414 m
  constexpr double min_w = 0.0;
  const double w_pose = std::clamp(1.0 - dist2 * d_max_square_inv, min_w, 1.0);

  // Blend position
  pred.pose.position.x = pred.pose.position.x * (1 - w_pose) + meas.pose.position.x * w_pose;
  pred.pose.position.y = pred.pose.position.y * (1 - w_pose) + meas.pose.position.y * w_pose;

  // Use smoothed shape and its area
  pred.shape = smoothed_shape;
  pred.area = types::getArea(smoothed_shape);

  // Blend orientation
  if (meas.kinematics.orientation_availability != types::OrientationAvailability::UNAVAILABLE) {
    double yaw_pred = tf2::getYaw(pred.pose.orientation);
    double yaw_meas = tf2::getYaw(meas.pose.orientation);

    // Handle SIGN_UNKNOWN: limit yaw difference to [-90°, 90°] to prevent sudden rotations
    if (meas.kinematics.orientation_availability == types::OrientationAvailability::SIGN_UNKNOWN) {
      double yaw_diff = yaw_meas - yaw_pred;
      // Normalize yaw_diff to [-π, π] using fmod
      yaw_diff = std::fmod(yaw_diff + M_PI, 2 * M_PI) - M_PI;
      if (yaw_diff > M_PI_2) {
        yaw_diff -= M_PI;
      } else if (yaw_diff < -M_PI_2) {
        yaw_diff += M_PI;
      }
      yaw_meas = yaw_pred + yaw_diff;
    }

    double yaw_fused = yaw_pred * (1 - w_pose) + yaw_meas * w_pose;
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw_fused);
    pred.pose.orientation = tf2::toMsg(q);
  }

  // Enlarge covariance if requested (for weak updates)
  if (enlarge_covariance) {
    using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    constexpr double additional_position_cov = 9.0;     // [m^2] additional variance
    constexpr double additional_orientation_cov = 0.5;  // [rad^2] additional variance
    constexpr double additional_velocity_cov = 25.0;    // [m^2/s^2] additional variance

    pred.pose_covariance[XYZRPY_COV_IDX::X_X] += additional_position_cov;
    pred.pose_covariance[XYZRPY_COV_IDX::Y_Y] += additional_position_cov;
    pred.pose_covariance[XYZRPY_COV_IDX::YAW_YAW] += additional_orientation_cov;

    // Enlarge velocity covariance if available
    if (pred.kinematics.has_twist_covariance) {
      pred.twist_covariance[XYZRPY_COV_IDX::X_X] += additional_velocity_cov;
      pred.twist_covariance[XYZRPY_COV_IDX::Y_Y] += additional_velocity_cov;
    }
  }

  return true;
}

void Tracker::updateClassification(
  const std::vector<autoware_perception_msgs::msg::ObjectClassification> & input)
{
  // classification algorithm:
  // 1. Update the matched classification probability
  // 2. If the label is not found, add it to the classification list
  // 3. Normalize tracking classification

  // If no existing classification, initialize with input
  if (classification_.empty()) {
    classification_ = input;
    return;
  }

  // Process existing classes
  for (auto & a_class : classification_) {
    // Find corresponding measurement
    auto it = std::find_if(input.begin(), input.end(), [&a_class](const auto & new_class) {
      return new_class.label == a_class.label;
    });

    if (it != input.end()) {
      // Class found in measurement
      constexpr float true_positive_rate = 0.8f;
      constexpr float false_positive_rate = 0.2f;
      a_class.probability = updateProbability(
        a_class.probability, it->probability * true_positive_rate, false_positive_rate);
    } else {
      // Class not observed in measurement
      constexpr float false_negative_rate = 0.6f;
      constexpr float true_negative_rate = 0.8f;
      constexpr float true_positive_rate = 1.0f - false_negative_rate;
      constexpr float false_positive_rate = 1.0f - true_negative_rate;
      a_class.probability =
        updateProbability(a_class.probability, true_positive_rate, false_positive_rate);
    }
  }

  // Add new classes from measurement that weren't in tracker
  for (const auto & new_class : input) {
    bool found = std::any_of(
      classification_.begin(), classification_.end(),
      [&new_class](const auto & old_class) { return old_class.label == new_class.label; });

    if (!found) {
      constexpr float true_positive_rate = 0.8f;
      auto adding_class = new_class;
      // New class gets probability weighted by measurement confidence
      adding_class.probability = new_class.probability * true_positive_rate;
      classification_.push_back(adding_class);
    }
  }

  // Normalization
  {
    float sum = 0.0;
    for (const auto & a_class : classification_) {
      sum += a_class.probability;
    }
    // Normalize only if the total probability is greater than 1.0
    if (sum > 1.0) {
      for (auto & a_class : classification_) {
        a_class.probability /= sum;
      }
    }
  }
}

uint Tracker::getChannelIndex() const
{
  // Return the index of the channel that has highest priority
  // lower the index, higher the priority

  uint index = types::max_channel_size - 1;  // Default to lowest priority index
  float max_probability = 0.0f;
  constexpr float threshold = 0.5;
  for (uint i = 0; i < existence_probabilities_.size(); ++i) {
    if (existence_probabilities_[i] > threshold) {
      return i;
    }
    if (existence_probabilities_[i] > max_probability) {
      max_probability = existence_probabilities_[i];
      index = i;
    }
  }
  // If no channel has a probability above the threshold, return the highest probability index
  return index;
}

void Tracker::limitObjectExtension(const object_model::ObjectModel object_model)
{
  auto & object_extension = object_.shape.dimensions;
  // set maximum and minimum size
  object_extension.x = std::clamp(
    object_extension.x, object_model.size_limit.length_min, object_model.size_limit.length_max);
  object_extension.y = std::clamp(
    object_extension.y, object_model.size_limit.width_min, object_model.size_limit.width_max);
  object_extension.z = std::clamp(
    object_extension.z, object_model.size_limit.height_min, object_model.size_limit.height_max);
}

void Tracker::getPositionCovarianceEigenSq(
  const rclcpp::Time & time, double & major_axis_sq, double & minor_axis_sq) const
{
  // estimate the covariance of the position at the given time
  types::DynamicObject object = object_;
  if (object.time.seconds() + 1e-6 < time.seconds()) {  // 1usec is allowed error
    getTrackedObject(time, object);
  }
  using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  auto & pose_cov = object.pose_covariance;

  // principal component of the position covariance matrix
  Eigen::Matrix2d covariance;
  covariance << pose_cov[XYZRPY_COV_IDX::X_X], pose_cov[XYZRPY_COV_IDX::X_Y],
    pose_cov[XYZRPY_COV_IDX::Y_X], pose_cov[XYZRPY_COV_IDX::Y_Y];
  // check if the covariance is valid
  if (covariance(0, 0) <= 0.0 || covariance(1, 1) <= 0.0) {
    RCLCPP_WARN(
      rclcpp::get_logger("Tracker"), "Covariance is not valid. X_X: %f, Y_Y: %f", covariance(0, 0),
      covariance(1, 1));
    major_axis_sq = 0.0;
    minor_axis_sq = 0.0;
    return;
  }
  // Direct eigenvalue calculation for 2x2 symmetric matrix
  const double a = covariance(0, 0);
  const double b = covariance(0, 1);
  const double c = covariance(1, 1);
  const double trace = a + c;
  const double det = a * c - b * b;
  const double sqrt_term = std::sqrt(trace * trace / 4.0 - det);

  major_axis_sq = trace / 2.0 + sqrt_term;
  minor_axis_sq = trace / 2.0 - sqrt_term;
}

double Tracker::getBEVArea() const
{
  const auto & dims = object_.shape.dimensions;
  return dims.x * dims.y;
}

double Tracker::getDistanceSqToEgo(const std::optional<geometry_msgs::msg::Pose> & ego_pose) const
{
  constexpr double INVALID_DISTANCE_SQ = -1.0;
  if (!ego_pose) {
    return INVALID_DISTANCE_SQ;
  }
  const auto & p = object_.pose.position;
  const auto & e = ego_pose->position;
  const double dx = p.x - e.x;
  const double dy = p.y - e.y;
  return dx * dx + dy * dy;
}

double Tracker::computeAdaptiveThreshold(
  double base_threshold, double fallback_threshold, const AdaptiveThresholdCache & cache,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose) const
{
  const double distance_sq = getDistanceSqToEgo(ego_pose);
  if (distance_sq < 0.0) return fallback_threshold;

  const double bev_area = getBEVArea();

  const double bev_area_influence = cache.getBEVAreaInfluence(bev_area);
  const double distance_influence = cache.getDistanceInfluence(distance_sq);

  return base_threshold + bev_area_influence + distance_influence;
}

bool Tracker::isConfident(
  const rclcpp::Time & time, const AdaptiveThresholdCache & cache,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose) const
{
  // check the number of measurements. if the measurement is too small, definitely not confident
  const int count = getTotalMeasurementCount();
  if (count < 2) {
    return false;
  }

  double major_axis_sq = 0.0;
  double minor_axis_sq = 0.0;
  getPositionCovarianceEigenSq(time, major_axis_sq, minor_axis_sq);

  // if the covariance is very small, the tracker is confident
  constexpr double STRONG_COV_THRESHOLD = 0.28;
  if (major_axis_sq < STRONG_COV_THRESHOLD) {
    return true;
  }

  // if the existence probability is high and the covariance is small enough with respect to its
  // distance to ego and its bev area, the tracker is confident
  // base threshold is 1.6, fallback threshold is 2.6;
  const double adaptive_threshold = computeAdaptiveThreshold(1.6, 2.6, cache, ego_pose);

  if (getTotalExistenceProbability() > 0.50 && major_axis_sq < adaptive_threshold) {
    return true;
  }

  return false;
}

bool Tracker::isExpired(
  const rclcpp::Time & now, const AdaptiveThresholdCache & cache,
  const std::optional<geometry_msgs::msg::Pose> & ego_pose) const
{
  // check the number of no measurements
  const double elapsed_time = getElapsedTimeFromLastUpdate(now);

  // if the last measurement is too old, the tracker is expired
  constexpr double EXPIRED_TIME_THRESHOLD = 1.0;  // [sec]
  if (elapsed_time > EXPIRED_TIME_THRESHOLD) {
    return true;
  }

  // if the tracker is not confident, the tracker is expired
  constexpr double EXPIRED_PROBABILITY_THRESHOLD = 0.015;
  const float existence_probability = getTotalExistenceProbability();
  if (existence_probability < EXPIRED_PROBABILITY_THRESHOLD) {
    return true;
  }

  // if the tracker is a bit old and the existence probability is low, check the covariance size
  constexpr double TIME_TO_CHECK_COV = 0.18;  // [sec]
  constexpr double EXISTENCE_PROBABILITY_TO_CHECK_COV = 0.3;
  if (
    elapsed_time > TIME_TO_CHECK_COV &&
    existence_probability < EXISTENCE_PROBABILITY_TO_CHECK_COV) {
    // if the tracker covariance is too large, the tracker is expired
    double major_axis_sq = 0.0;
    double minor_axis_sq = 0.0;
    getPositionCovarianceEigenSq(now, major_axis_sq, minor_axis_sq);
    // major_cov: base_threshold is 2.8, fallback threshold is 3.8;
    // minor_cov: base_threshold is 2.7, fallback threshold is 3.7;
    const double major_cov_threshold = computeAdaptiveThreshold(2.8, 3.8, cache, ego_pose);
    const double minor_cov_threshold = computeAdaptiveThreshold(2.7, 3.7, cache, ego_pose);
    if (major_axis_sq > major_cov_threshold || minor_axis_sq > minor_cov_threshold) {
      return true;
    }
  }

  return false;
}

float Tracker::getKnownObjectProbability() const
{
  // find unknown probability
  float unknown_probability = 0.0;
  for (const auto & a_class : object_.classification) {
    if (a_class.label == autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) {
      unknown_probability = a_class.probability;
      break;
    }
  }
  // known object probability is reverse of unknown probability
  return 1.0 - unknown_probability;
}

double Tracker::getPositionCovarianceDeterminant() const
{
  using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  auto & pose_cov = object_.pose_covariance;

  // The covariance size is defined as the square of the dominant eigenvalue
  // of the 2x2 covariance matrix:
  // | X_X  X_Y |
  // | Y_X  Y_Y |
  const double determinant = pose_cov[XYZRPY_COV_IDX::X_X] * pose_cov[XYZRPY_COV_IDX::Y_Y] -
                             pose_cov[XYZRPY_COV_IDX::X_Y] * pose_cov[XYZRPY_COV_IDX::Y_X];
  // covariance matrix is positive semi-definite
  if (determinant <= 0.0) {
    RCLCPP_WARN(
      rclcpp::get_logger("Tracker"), "Covariance is not positive semi-definite. X_X: %f, Y_Y: %f",
      pose_cov[XYZRPY_COV_IDX::X_X], pose_cov[XYZRPY_COV_IDX::Y_Y]);
    // return a large value to indicate the covariance is not valid
    return std::numeric_limits<double>::max();
  }
  return determinant;
}

bool Tracker::conditionedUpdate(
  const types::DynamicObject & measurement, const types::DynamicObject & prediction,
  const autoware_perception_msgs::msg::Shape & smoothed_shape,
  const rclcpp::Time & measurement_time, const types::InputChannel & channel_info,
  std::string & update_strategy, bool is_debug_target)
{
  if (is_debug_target) {
    std::cout << "DEBUG: Tracker::conditionedUpdate() (BASE CLASS) called!" << std::endl;
  }

  // For non-vehicle trackers, create pseudo measurement
  types::DynamicObject pseudo_measurement = prediction;
  createPseudoMeasurement(measurement, pseudo_measurement, smoothed_shape);

  // Apply the measurement update directly
  measure(pseudo_measurement, measurement_time, channel_info);

  update_strategy = "NON_VEHICLE";

  return true;
}

}  // namespace autoware::multi_object_tracker
