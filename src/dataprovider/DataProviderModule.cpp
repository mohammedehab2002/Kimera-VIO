/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   DataProviderModule.cpp
 * @brief  Pipeline Module that takes care of providing data to the VIO
 * pipeline.
 * @author Antoni Rosinol
 */

#include "kimera-vio/dataprovider/DataProviderModule.h"

namespace VIO {

DataProviderModule::DataProviderModule(OutputQueue* output_queue,
                                       const std::string& name_id,
                                       const bool& parallel_run)
    : MISO(output_queue, name_id, parallel_run),
      imu_data_(),
      // not super nice to init a member with another member in ctor...
      timestamp_last_frame_(kNoFrameYet),
      do_initial_imu_timestamp_correction_(false),
      imu_timestamp_correction_(0),
      imu_time_shift_ns_(0) {}

bool DataProviderModule::getTimeSyncedImuMeasurements(
    const Timestamp& timestamp,
    ImuMeasurements* imu_meas) {
  CHECK_NOTNULL(imu_meas);
  CHECK_LT(timestamp_last_frame_, timestamp)
      << "Timestamps out of order:\n"
      << " - Last Frame Timestamp = " << timestamp_last_frame_ << '\n'
      << " - Current Timestamp = " << timestamp;

  if (imu_data_.imu_buffer_.size() == 0) {
    VLOG(1) << "No IMU measurements available yet, dropping this frame.";
    return false;
  }

  // Extract imu measurements between consecutive frames.
  if (timestamp_last_frame_ == kNoFrameYet) {
    // TODO(Toni): wouldn't it be better to get all IMU measurements up to
    // this
    // timestamp? We should add a method to the IMU buffer for that.
    VLOG(1) << "Skipping first frame, because we do not have a concept of "
               "a previous frame timestamp otherwise.";
    timestamp_last_frame_ = timestamp;
    return false;
  }

  // Do a very coarse timestamp correction to make sure that the IMU data
  // is aligned enough to send packets to the front-end. This is assumed
  // to be very inaccurate and should not be enabled without some other
  // actual time alignment in the frontend
  // if (false) {
  if (do_initial_imu_timestamp_correction_) {
    CHECK_GT(imu_data_.imu_buffer_.size(), 0)
        << "IMU buffer lost measurements unexpectedly";
    ImuMeasurement newest_imu;
    imu_data_.imu_buffer_.getNewestImuMeasurement(&newest_imu);
    // this is delta = imu.timestamp - frame.timestamp so that when querying,
    // we get query = new_frame.timestamp + delta = frame_delta + imu.timestamp
    imu_timestamp_correction_ = newest_imu.timestamp_ - timestamp;
    if (imu_data_.imu_rate_ != 0.0) {
      // If the timestamp difference is small enough to be explained by a
      // sampling difference (i.e. that the timestamps are within one IMU
      // period), then force the coarse alignment to 0
      const double imu_period = 1.0 / imu_data_.imu_rate_;
      imu_timestamp_correction_ =
          std::abs(imu_timestamp_correction_) < imu_period
              ? 0.0
              : imu_timestamp_correction_;
    }
    do_initial_imu_timestamp_correction_ = false;
    VLOG(1) << "Computed intial time alignment of "
            << imu_timestamp_correction_;
  }

  // imu_time_shift_ can be externally, asynchronously modified, which would
  // be bad if it happended during the loop. Caching here prevents this nasty
  // race condition and avoids locking for the entire call to
  // getTimeSyncedImuMeasurements.
  // Also note that the imu_timestamp_correction_ doesn't currently do anything
  // here unless do_initial_imu_timestamp_correction_ is enabled
  const Timestamp curr_imu_time_shift = imu_time_shift_ns_;
  utils::ThreadsafeImuBuffer::QueryResult query_result =
      utils::ThreadsafeImuBuffer::QueryResult::kDataNeverAvailable;
  bool log_error_once = true;

  // Note that the second term (-t_frame_start + t_imu_start) is a coarse
  // correction to provide the timestamp of the imu measurements in the "image
  // timing coordinate frame" and the t_imu_from_cam is the transform to the imu
  // timing coordinate frame
  /* t_last_imu = t_last_frame + (-t_frame_start + t_imu_start) +
   * (t_imu_from_cam) */
  /* t_curr_imu = t_curr_frame + (-t_frame_start + t_imu_start) +
   * (t_imu_from_cam) */
  while (
      !MISO::shutdown_ &&
      (query_result = imu_data_.imu_buffer_.getImuDataInterpolatedUpperBorder(
           timestamp_last_frame_ + imu_timestamp_correction_ +
               curr_imu_time_shift,
           timestamp + imu_timestamp_correction_ + curr_imu_time_shift,
           &imu_meas->timestamps_,
           &imu_meas->acc_gyr_)) !=
          utils::ThreadsafeImuBuffer::QueryResult::kDataAvailable) {
    VLOG(1) << "No IMU data available. Reason:\n";
    switch (query_result) {
      case utils::ThreadsafeImuBuffer::QueryResult::kDataNotYetAvailable: {
        if (log_error_once) {
          LOG(WARNING) << "Waiting for IMU data...";
          log_error_once = false;
        }
        continue;
      }
      case utils::ThreadsafeImuBuffer::QueryResult::kQueueShutdown: {
        LOG(WARNING)
            << "IMU buffer was shutdown. Shutting down DataProviderModule.";
        MISO::shutdown();
        return false;
      }
      case utils::ThreadsafeImuBuffer::QueryResult::kDataNeverAvailable: {
        LOG(WARNING)
            << "Asking for data before start of IMU stream, from timestamp: "
            << timestamp_last_frame_ << " to timestamp: " << timestamp;
        // Ignore frames that happened before the earliest imu data
        timestamp_last_frame_ = timestamp;
        return false;
      }
      case utils::ThreadsafeImuBuffer::QueryResult::
          kTooFewMeasurementsAvailable: {
        LOG(WARNING) << "No IMU measurements here, and IMU data stream already "
                        "passed this time region"
                     << "from timestamp: " << timestamp_last_frame_
                     << " to timestamp: " << timestamp;
        return false;
      }
      case utils::ThreadsafeImuBuffer::QueryResult::kDataAvailable: {
        LOG(FATAL) << "We should not be inside this while loop if IMU data is "
                      "available...";
        return false;
      }
    }
  }
  timestamp_last_frame_ = timestamp;

  // adjust the timestamps for the frontend
  // TODO(nathan) may also need to apply imu_time_shift_ here
  imu_meas->timestamps_.array() -= imu_timestamp_correction_;

  VLOG(10) << "////////////////////////////////////////// Creating packet!\n"
           << "STAMPS IMU rows : \n"
           << imu_meas->timestamps_.rows() << '\n'
           << "STAMPS IMU cols : \n"
           << imu_meas->timestamps_.cols() << '\n'
           << "STAMPS IMU: \n"
           << imu_meas->timestamps_ << '\n'
           << "ACCGYR IMU rows : \n"
           << imu_meas->acc_gyr_.rows() << '\n'
           << "ACCGYR IMU cols : \n"
           << imu_meas->acc_gyr_.cols() << '\n'
           << "ACCGYR IMU: \n"
           << imu_meas->acc_gyr_;

  return true;
}

void DataProviderModule::shutdownQueues() {
  imu_data_.imu_buffer_.shutdown();
  MISO::shutdownQueues();
}

}  // namespace VIO
