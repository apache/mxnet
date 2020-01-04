/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
 
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <Windows.h>
#include <io.h>
#include <vector>
#include <regex>
#include <shlwapi.h>


extern "C" IMAGE_DOS_HEADER __ImageBase;


std::vector<int> find_mxnet_dll()
{
	std::vector<int> version;
	intptr_t handle;

	_wfinddata_t findData{};
	std::wregex reg(L".*?mxnet_([0-9]+)\\.dll");

	HMODULE hModule = reinterpret_cast<HMODULE>(&__ImageBase);
	WCHAR szPathBuffer[MAX_PATH] = { 0 };
	GetModuleFileNameW(hModule, szPathBuffer, MAX_PATH);

	PathRemoveFileSpecW(szPathBuffer);
	wcscat_s(szPathBuffer, L"\\mxnet_*.dll");

	handle = _wfindfirst(szPathBuffer, &findData);
	if (handle == -1)
	{
		return version;
	}

	do
	{
		if (!(findData.attrib & _A_SUBDIR) || wcscmp(findData.name, L".") != 0 || wcscmp(findData.name, L"..") != 0)
		{
			std::wstring str(findData.name);
			std::wsmatch base_match;
			if(std::regex_match(str, base_match, reg))
			{
				if (base_match.size() == 2) {
					std::wssub_match base_sub_match = base_match[1];
					std::wstring base = base_sub_match.str();
					version.push_back(std::stoi(base)) ;
				}
			}
		}
	} while (_wfindnext(handle, &findData) == 0);    

	_findclose(handle);
	std::sort(version.begin(), version.end());
	return version;
}

int find_version()
{
	std::vector<int> known_sm = find_mxnet_dll();
	int count = 0;
	int version = 75;
	if (cudaSuccess != cudaGetDeviceCount(&count))
	{
		return 30;
	}
	if (count == 0)
	{
		return 30;
	}


	for (int device = 0; device < count; ++device)
	{
		cudaDeviceProp prop{};
		if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
		{
			version = std::min(version, prop.major * 10 + prop.minor);
		}
	}

	for (int i = known_sm.size() -1 ; i >=0; --i)
	{
		if(known_sm[i]<= version)
		{
			return known_sm[i];
		}
	}

	return version;
}

void load_function(HMODULE hm);

void mxnet_init()
{
	int version = find_version();
	WCHAR dll_name[MAX_PATH];
	wsprintfW(dll_name, L"mxnet_%d.dll", version);
	HMODULE hm = LoadLibraryW(dll_name);
	load_function(hm);
}


extern "C" BOOL WINAPI DllMain(
	HINSTANCE const instance,  // handle to DLL module
	DWORD     const reason,    // reason for calling function
	LPVOID    const reserved)  // reserved
{
	// Perform actions based on the reason for calling.
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
		mxnet_init();
		// Initialize once for each new process.
		// Return FALSE to fail DLL load.
		break;

	case DLL_THREAD_ATTACH:
		// Do thread-specific initialization.
		break;

	case DLL_THREAD_DETACH:
		// Do thread-specific cleanup.
		break;

	case DLL_PROCESS_DETACH:
		// Perform any necessary cleanup.
		break;
	}
	return TRUE;  // Successful DLL_PROCESS_ATTACH.
}