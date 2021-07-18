package org.apache.mxnet.engine;

public class DeviceType {

    private static final String CPU_PINNED = "cpu_pinned";

    private DeviceType() {}

    /**
     * Converts a {@link Device} to the corresponding MXNet device number.
     *
     * @param device the java {@link Device}
     * @return the MXNet device number
     * @exception IllegalArgumentException the device is null or is not supported
     */
    public static int toDeviceType(Device device) {
        if (device == null) {
            throw new IllegalArgumentException("Unsupported device: null");
        }

        String deviceType = device.getDeviceType();

        if (Device.Type.CPU.equals(deviceType)) {
            return 1;
        } else if (Device.Type.GPU.equals(deviceType)) {
            return 2;
        } else if (CPU_PINNED.equals(deviceType)) {
            return 3;
        } else {
            throw new IllegalArgumentException("Unsupported device: " + device.toString());
        }
    }

    /**
     * Converts from an MXNet device number to {@link Device}.
     *
     * @param deviceType the MXNet device number
     * @return the corresponding {@link Device}
     */
    public static String fromDeviceType(int deviceType) {
        switch (deviceType) {
            case 1:
            case 3:
                // hide the CPU_PINNED to frontend user
                // but the advance user can still create CPU_PINNED
                // to pass through engine
                return Device.Type.CPU;
            case 2:
                return Device.Type.GPU;
            default:
                throw new IllegalArgumentException("Unsupported deviceType: " + deviceType);
        }
    }

}
