#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>

typedef std::vector<unsigned char> vchar;

template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

vchar getDeviceProperty(cl_device_id device, cl_device_info param_name)
{
    size_t propertySize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &propertySize));
    vchar property(propertySize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, propertySize, property.data(), nullptr));
    return property;
}

cl_ulong getDevicePropertyValue(cl_device_id device, cl_device_info param_name)
{
    size_t propertySize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &propertySize));
    cl_ulong property = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, propertySize, &property, nullptr));
    return property;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте 
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.
        try {
            std::size_t randomNumber = 239;
            OCL_SAFE_CALL(clGetPlatformInfo(platform, randomNumber, 0, nullptr, &platformNameSize));
            // Expected:
            // retcode = -30
            // #define CL_INVALID_VALUE                            -30
            // CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.
        }
        catch (const std::runtime_error& e) {
            std::cerr << "- TODO 1.1: try to call clGetPlatformInfo() with incorrect param_name" << std::endl;
            std::cerr << "    " << e.what() << std::endl;
        }

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        vchar platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "- TODO 1.2: get Platform name" << std::endl;
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        vchar platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
        std::cout << "- TODO 1.3: get Platform vendor" << std::endl;
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "- TODO 2.1: devices count" << std::endl;
        std::cout << "    Platform devices count: " << devicesCount << std::endl;

        std::cout << "- TODO 2.2: devices info" << std::endl;
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::cout << "  Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            vchar deviceName = getDeviceProperty(device, CL_DEVICE_NAME);
            cl_device_type deviceType = getDevicePropertyValue(device, CL_DEVICE_TYPE);
            cl_ulong memSizeMb = getDevicePropertyValue(device, CL_DEVICE_GLOBAL_MEM_SIZE) >> 20; // bytes -> Mb
            vchar deviceVendor = getDeviceProperty(device, CL_DEVICE_VENDOR);
            vchar openclVersion = getDeviceProperty(device, CL_DEVICE_OPENCL_C_VERSION);
            vchar deviceProfile = getDeviceProperty(device, CL_DEVICE_PROFILE);

            std::cout << "    device name: " << deviceName.data() << std::endl;
            std::cout << "    device type: " << deviceType << std::endl;
            std::cout << "    memory size [Mb]: " << memSizeMb << std::endl;
            std::cout << "    device vendor: " << deviceVendor.data() << std::endl;
            std::cout << "    OpenCL version: " << openclVersion.data() << std::endl;
            std::cout << "    device profile: " << deviceProfile.data() << std::endl;
        }
    }

    return 0;
}
