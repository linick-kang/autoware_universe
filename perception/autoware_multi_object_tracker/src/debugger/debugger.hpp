// Copyright 2024 TIER IV, Inc.
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

#ifndef DEBUGGER__DEBUGGER_HPP_
#define DEBUGGER__DEBUGGER_HPP_

#include "autoware/multi_object_tracker/object_model/types.hpp"
#include "debug_object.hpp"

#include <autoware_utils/ros/debug_publisher.hpp>
#include <autoware_utils/ros/published_time_publisher.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <diagnostic_updater/publisher.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::multi_object_tracker
{

/**
 * @brief Debugger class for multi object tracker
 * @details This class is used to publish debug information of multi object tracker
 */
class TrackerDebugger
{
public:
  TrackerDebugger(
    rclcpp::Node & node, const std::string & frame_id,
    const std::vector<types::InputChannel> & channels_config);

private:
  // Timing check utilities
  struct TimingCheckResult
  {
    std::string message;
    diagnostic_msgs::msg::DiagnosticStatus::_level_type level;
  };
  TimingCheckResult checkDelayTiming(double delay) const;
  TimingCheckResult checkExtrapolationTiming(
    double extrapolation_time, const rclcpp::Time & timestamp);
  static TimingCheckResult determineOverallTimingStatus(
    bool no_published_trackers, const TimingCheckResult & delay_result,
    const TimingCheckResult & extrapolation_result);
  // Debug settings
  struct DEBUG_SETTINGS
  {
    bool publish_processing_time;
    bool publish_tentative_objects;
    bool publish_debug_markers;
    double diagnostics_warn_delay;
    double diagnostics_error_delay;
    double diagnostics_warn_extrapolation;
    double diagnostics_error_extrapolation;
  } debug_settings_;

  // Diagnostic values
  struct DiagnosticValues
  {
    double min_extrapolation_time = 0.0;
    size_t published_trackers_count = 0;
  } diagnostic_values_;

  // ROS node, publishers
  rclcpp::Node & node_;
  rclcpp::Publisher<autoware_perception_msgs::msg::TrackedObjects>::SharedPtr
    debug_tentative_objects_pub_;
  std::unique_ptr<autoware_utils::DebugPublisher> processing_time_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_objects_markers_pub_;

  diagnostic_updater::Updater diagnostic_updater_;
  // Object debugger
  TrackerObjectDebugger object_debugger_;
  // Time measurement
  bool is_initialized_ = false;
  double pipeline_latency_ms_ = 0.0;
  rclcpp::Time last_input_stamp_;
  rclcpp::Time stamp_process_start_;
  rclcpp::Time stamp_process_end_;
  rclcpp::Time stamp_publish_start_;
  rclcpp::Time stamp_publish_output_;
  rclcpp::Time last_non_warning_timestamp_;

  // Configuration
  void setupDiagnostics();
  void loadParameters();

public:
  // Single update method for all diagnostic values
  void updateDiagnosticValues(double min_extrapolation_time, size_t published_count);

  // Object publishing
  bool shouldPublishTentativeObjects() const { return debug_settings_.publish_tentative_objects; }
  void publishTentativeObjects(
    const autoware_perception_msgs::msg::TrackedObjects & tentative_objects) const;

  // Time measurement
  void startMeasurementTime(
    const rclcpp::Time & now, const rclcpp::Time & measurement_header_stamp);
  void endMeasurementTime(const rclcpp::Time & now);
  void startPublishTime(const rclcpp::Time & now);
  void endPublishTime(const rclcpp::Time & now, const rclcpp::Time & object_time);
  // cppcheck-suppress functionConst
  void checkAllTiming(diagnostic_updater::DiagnosticStatusWrapper & stat);
  // Debug object
  void collectObjectInfo(
    const rclcpp::Time & message_time, const std::list<std::shared_ptr<Tracker>> & list_tracker,
    const types::DynamicObjectList & detected_objects,
    const std::unordered_map<int, int> & direct_assignment,
    const std::unordered_map<int, int> & reverse_assignment);
  void publishObjectsMarkers();
};

}  // namespace autoware::multi_object_tracker

#endif  // DEBUGGER__DEBUGGER_HPP_
