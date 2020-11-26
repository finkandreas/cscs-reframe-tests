#ifndef __INCLUDED_CUDA_TOOLS__
#define __INCLUDED_CUDA_TOOLS__

#include <iostream>
#include <unistd.h>
#include <nvml.h>

static inline void nvmlCheck(nvmlReturn_t err)
{
# ifdef DEBUG
  if (err != NVML_SUCCESS)
  {
    std::cerr << "Call to the nvml API failed!" << std::endl;
    exit(1);
  }
# endif
}

Smi::Smi()
{
  if (!(this->smiIsActive))
  {
    nvmlCheck( nvmlInit() );
    this->smiIsActive = 1;
    nvmlCheck( nvmlDeviceGetCount(&numberOfDevices) );
  }

  this->activeSmiInstances += 1;
}

void Smi::setCpuAffinity(int id)
{
  checkGpuIdIsSensible(id);

  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );
  nvmlCheck( nvmlDeviceSetCpuAffinity(device) );
}

float Smi::getGpuTemp(int id)
{
  // Check that the gpu id is sensible
  checkGpuIdIsSensible(id);

  // Get device handle
  nvmlDevice_t device;
  nvmlCheck( nvmlDeviceGetHandleByIndex(id, &device) );

  // Get the temperature
  unsigned int temperature;
  nvmlCheck( nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature) );
  return float(temperature);
}

Smi::~Smi()
{
  this->activeSmiInstances -= 1;
  if (!(this->activeSmiInstances))
  {
    nvmlCheck( nvmlShutdown() );
    this->nvmlIsActive = 0;
  }
}

#endif
