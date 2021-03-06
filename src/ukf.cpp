#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Debug purpose Eigen Matrix format
Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;

    /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
   * End DO NOT MODIFY section for measurement noise values 
   */

    /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

    is_initialized_ = false;
    time_us_ = 0;

    // State dimensions: px, py, speed, yaw, yaw rate
    n_x_ = 5;

    // initial state vector
    x_ = VectorXd(n_x_);
    x_.fill(0.0);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0.0225, 0,
        0, 0, 0, 0, .0225;

    // Augmented state dimensions: state dimetnions + lateral acceleration nose and yaw acceleration noise
    n_aug_ = 7;
    lambda_ = 3 - n_aug_;
    sigma_points_num_ = 2 * n_aug_ + 1;

    weights_ = Eigen::VectorXd(sigma_points_num_);
    // Initializing weights of sigma points
    for (int i = 0; i < sigma_points_num_; i++) {
        if (i == 0) {
            weights_(i) = lambda_ / (lambda_ + n_aug_);
        } else {
            weights_(i) = 0.5 / (n_aug_ + lambda_);
        }
    }

    // Sigma points vector
    Xsig_pred_ = MatrixXd(n_x_, sigma_points_num_);
}

UKF::~UKF() { }

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

    if (!is_initialized_) {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            // Initialize for radar measurement
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            double rho_dot = meas_package.raw_measurements_(3);

            // Converting radar measurement to CTRV model state space
            double px = rho * cos(phi);
            double py = rho * sin(phi);

            x_ << px,
                py,
                0,
                0,
                0;

        } else {
            // Initialize for lidar measurement
            x_ << meas_package.raw_measurements_(0),
                meas_package.raw_measurements_(1),
                0,
                0,
                0;
        }

        time_us_ = meas_package.timestamp_;

        is_initialized_ = true;
        std::cout << "Initialized Unscented Kalman Filter" << std::endl;
        return;
    }

    /*
        Predict
    */

    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(delta_t);

    /*
        Update
   */

    Update(meas_package);
}

void UKF::Prediction(double delta_t)
{
    // Augmented state vector
    Eigen::VectorXd x_aug_ = Eigen::VectorXd(n_aug_);
    // Augmented process covariance
    Eigen::MatrixXd P_aug_ = Eigen::MatrixXd(n_aug_, n_aug_);

    x_aug_.head(5) = x_;
    x_aug_(5) = 0.0; // longitudinal acceleration noise
    x_aug_(6) = 0.0; // yaw acceleration noise

    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(5, 5) = P_;
    P_aug_(5, 5) = std_a_ * std_a_;
    P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

    // Choose the sigma points
    Eigen::MatrixXd X_sig_aug_ = Eigen::MatrixXd(n_aug_, sigma_points_num_);

    MatrixXd A = P_aug_.llt().matrixL(); // calculate square root of P

    // initializing augmented sigma points
    X_sig_aug_.col(0) = x_aug_; // first sigma point is the mean

    for (int i = 0; i < n_aug_; i++) {
        X_sig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);
        X_sig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
    }

    // Predict the sigma points
    for (int i = 0; i < sigma_points_num_; i++) {
        double px = X_sig_aug_(0, i);
        double py = X_sig_aug_(1, i);
        double speed = X_sig_aug_(2, i);
        double yaw = X_sig_aug_(3, i);
        double yaw_rate = X_sig_aug_(4, i);
        double nu_a = X_sig_aug_(5, i);
        double nu_yawdd = X_sig_aug_(6, i);

        // Predicted states
        double px_p, py_p, speed_p, yaw_p, yaw_rate_p;

        // Apply the process model on the
        // check for zero yaw rate
        if (yaw_rate < 0.0001) {
            px_p = px + speed * cos(yaw) * delta_t;
            py_p = py + speed * sin(yaw) * delta_t;
        } else {
            double speed_to_yaw_rate = speed / yaw_rate;
            px_p = px + speed_to_yaw_rate * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
            py_p = py + speed_to_yaw_rate * (-1 * cos(yaw + yaw_rate * delta_t) + cos(yaw));
        }
        speed_p = speed;
        yaw_p = yaw + yaw_rate * delta_t;
        yaw_rate_p = yaw_rate;

        // apply noise
        px_p += 0.5 * pow(delta_t, 2) * cos(yaw) * nu_a;
        py_p += 0.5 * pow(delta_t, 2) * sin(yaw) * nu_a;
        speed_p += delta_t * nu_a;
        yaw_p += 0.5 * pow(delta_t, 2) * nu_yawdd;
        yaw_rate_p += delta_t * nu_yawdd;

        // Add predicted values to predicted sigma points matrix
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = speed_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yaw_rate_p;
    }
    // Calculate the prediction mean and covariance
    Eigen::VectorXd x_predicted_ = Eigen::VectorXd(n_x_);
    Eigen::MatrixXd P_predicted_ = Eigen::MatrixXd(n_x_, n_x_);

    x_predicted_.fill(0.0);
    P_predicted_.fill(0.0);

    for (int i = 0; i < sigma_points_num_; i++) {
        x_predicted_ = x_predicted_ + weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < sigma_points_num_; i++) {
        Eigen::VectorXd x_diff_ = Xsig_pred_.col(i) - x_predicted_;

        // angle normalization
        while (x_diff_(3) > M_PI)
            x_diff_(3) -= 2. * M_PI;
        while (x_diff_(3) < -M_PI)
            x_diff_(3) += 2. * M_PI;

        P_predicted_ = P_predicted_ + weights_(i) * x_diff_ * x_diff_.transpose();
    }

    // Update vector x_
    x_ = x_predicted_;
    P_ = P_predicted_;
}

void UKF::Update(MeasurementPackage meas_package)
{
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    } else {
        UpdateLidar(meas_package);
    }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    std::uint8_t n_z_ = 2; // number of state dimetnions: px, py
    Eigen::MatrixXd H_ = Eigen::MatrixXd(n_z_, n_x_); // measurement matrix
    Eigen::MatrixXd R_ = Eigen::MatrixXd(n_z_, n_z_); // measurement noise covariance matrix

    Eigen::VectorXd z_ = Eigen::VectorXd(n_z_); // measurement vector
    Eigen::MatrixXd S_ = Eigen::MatrixXd(n_z_, n_z_); // measurement covariance
    Eigen::MatrixXd K_ = Eigen::MatrixXd(n_z_, n_z_); // Kalman gain

    Eigen::VectorXd y_ = Eigen::VectorXd(n_z_); // error

    Eigen::MatrixXd I_ = Eigen::MatrixXd::Identity(n_x_, n_x_);

    H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

    R_ << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;

    z_ = meas_package.raw_measurements_;

    y_ = z_ - H_ * x_;
    S_ = H_ * P_ * H_.transpose() + R_;

    K_ = P_ * H_.transpose() * S_.inverse();

    x_ = x_ + K_ * y_;
    P_ = (I_ - K_ * H_) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    std::uint8_t n_z_ = 3; // number of state dimetnions : rho, phi, rho dot
    Eigen::MatrixXd R_ = Eigen::MatrixXd(n_z_, n_z_); // measurement noise covariance matrix
    Eigen::MatrixXd Z_sigma_points_ = Eigen::MatrixXd(n_z_, sigma_points_num_);
    Eigen::VectorXd z_ = Eigen::VectorXd(n_z_); // measurement mean vector
    Eigen::MatrixXd S_ = Eigen::MatrixXd(n_z_, n_z_); // measurement covariance

    Eigen::MatrixXd T_ = Eigen::MatrixXd(n_x_, n_z_); // cross corelation between sigma points in state space and measurement space

    R_ << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;

    z_.fill(0.0);
    S_.fill(0.0);
    T_.fill(0.0);

    // Apply the measurement function on predicted sigma points
    Z_sigma_points_.fill(0.0);
    for (int i = 0; i < sigma_points_num_; i++) {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double speed = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double vx = speed * cos(yaw);
        double vy = speed * sin(yaw);

        px = (fabs(px) <= 0.0001) ? 0.0001 : px;
        py = (fabs(py) <= 0.0001) ? 0.0001 : py;

        Z_sigma_points_(0, i) = sqrt(px * px + py * py) > 0.0001 ? sqrt(px * px + py * py) : 0.0001;
        Z_sigma_points_(1, i) = atan2(py, px);
        Z_sigma_points_(2, i) = (px * vx + py * vy) / sqrt(px * px + py * py);
    }

    // Predict the measurement mean
    for (int i = 0; i < sigma_points_num_; i++) {
        z_ = z_ + weights_(i) * Z_sigma_points_.col(i);
    }

    // Predict the measurement covariance
    for (int i = 0; i < sigma_points_num_; i++) {
        Eigen::VectorXd z_diff = Z_sigma_points_.col(i) - z_;
        S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
    }
    S_ = S_ + R_;

    // Calculate cross corelation between sigma points in state space and measurement space
    for (int i = 0; i < sigma_points_num_; i++) {
        Eigen::VectorXd z_diff = Z_sigma_points_.col(i) - z_;
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        T_ = T_ + weights_(i) * x_diff * z_diff.transpose();
    }

    // Update x_ and P_
    Eigen::MatrixXd K_ = T_ * S_.inverse();

    Eigen::VectorXd z_diff = meas_package.raw_measurements_ - z_;
    while (z_diff(1) > M_PI)
        z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
        z_diff(1) += 2. * M_PI;

    x_ = x_ + K_ * z_diff;
    P_ = P_ - K_ * S_ * K_.transpose();
}
