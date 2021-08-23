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
 
#include <iostream>  
#include <io.h>  
#include <Windows.h>  
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define  IMAGE_SIZEOF_SIGNATURE 4


DWORD rva_to_foa(IN DWORD RVA, IN PIMAGE_SECTION_HEADER section_header) 
{

	size_t count = 0;
	for (count = 1; RVA > (section_header->VirtualAddress + section_header->Misc.VirtualSize); count++, section_header++);

	DWORD FOA = RVA - section_header->VirtualAddress + section_header->PointerToRawData;

	return FOA;
} 

std::string format(const char* format, ...)
{
	va_list args;
	va_start(args, format);
#ifndef _MSC_VER
	size_t size = std::snprintf(nullptr, 0, format, args) + 1; // Extra space for '\0'
	std::unique_ptr<char[]> buf(new char[size]);
	std::vsnprintf(buf.get(), size, format, args);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
#else
	int size = _vscprintf(format, args) +1;
	std::unique_ptr<char[]> buf(new char[size]);
	vsnprintf_s(buf.get(), size, _TRUNCATE, format, args);
	return std::string(buf.get());
#endif
	va_end(args);
}

int main(int argc, char* argv[])
{

	if (argc != 2)
	{
		return 0;
	}

	//open file
	const HANDLE h_file = CreateFile(
		argv[1],   
		GENERIC_READ , 
		FILE_SHARE_READ ,
		nullptr,
		OPEN_EXISTING,  
		FILE_ATTRIBUTE_NORMAL,
		nullptr);


	DWORD size_high;
	const DWORD size_low = GetFileSize(h_file, &size_high);

	uint64_t dll_size = ((uint64_t(size_high)) << 32) + (uint64_t)size_low;

	// Create File Mapping
	const HANDLE h_map_file = CreateFileMapping(
		h_file,
		nullptr,
		PAGE_READONLY, 
		size_high,
		size_low,   
		nullptr);
	if (h_map_file == INVALID_HANDLE_VALUE || h_map_file == nullptr)
	{
		std::cout << "error";
		CloseHandle(h_file);
		return 0;
	}

	//Map File to memory
	void* pv_file = MapViewOfFile(
		h_map_file,
		FILE_MAP_READ,
		0,
		0,
		0);

	if (pv_file == nullptr)
	{
		std::cout << "error";
		CloseHandle(h_file);
		return 0;
	}

	uint8_t* p = static_cast<uint8_t*>(pv_file);


	PIMAGE_DOS_HEADER dos_header = reinterpret_cast<PIMAGE_DOS_HEADER>(p);

	const PIMAGE_NT_HEADERS nt_headers = reinterpret_cast<const PIMAGE_NT_HEADERS>(p + dos_header->e_lfanew);

	const PIMAGE_FILE_HEADER file_header = &nt_headers->FileHeader;

	PIMAGE_OPTIONAL_HEADER	optional_header = (PIMAGE_OPTIONAL_HEADER)(&nt_headers->OptionalHeader);

	const DWORD file_alignment = optional_header->FileAlignment;


	PIMAGE_SECTION_HEADER section_table =
		reinterpret_cast<PIMAGE_SECTION_HEADER>(p + dos_header->e_lfanew +
			IMAGE_SIZEOF_SIGNATURE +
			IMAGE_SIZEOF_FILE_HEADER +
			file_header->SizeOfOptionalHeader);

	DWORD	export_foa = rva_to_foa(optional_header->DataDirectory[0].VirtualAddress, section_table);

	PIMAGE_EXPORT_DIRECTORY export_directory = (PIMAGE_EXPORT_DIRECTORY)(p + export_foa);


	DWORD name_list_foa = rva_to_foa(export_directory->AddressOfNames, section_table);

	PDWORD name_list = (PDWORD)(p + name_list_foa);




	std::vector<std::string> func_list;

	for (size_t i = 0; i < export_directory->NumberOfNames; i++, name_list++)
	{

		DWORD name_foa = rva_to_foa(* name_list, section_table);
		char* name = (char*)(p + name_foa);
		func_list.emplace_back(name);

	}


	UnmapViewOfFile(pv_file); 
	CloseHandle(h_map_file);
	CloseHandle(h_file); 


	std::ofstream gen_cpp_obj;
	gen_cpp_obj.open("warp_gen_cpp.cpp", std::ios::out | std::ios::trunc);
	gen_cpp_obj << "#include <Windows.h>\n";
	gen_cpp_obj << "extern \"C\" \n{\n";


	for (size_t i = 0; i < func_list.size(); ++i)
	{
		auto fun = func_list[i];
		gen_cpp_obj << format("void * warp_point_%d;\n", i);
		gen_cpp_obj << format("#pragma comment(linker, \"/export:%s=warp_func_%d\")\n", fun.c_str(), i);
		gen_cpp_obj << format("void warp_func_%d();\n", i);
		gen_cpp_obj << ("\n");
	}
	gen_cpp_obj << ("}\n");


	gen_cpp_obj << ("void load_function(HMODULE hm)\n{\n");
	for (size_t i = 0; i < func_list.size(); ++i)
	{
		auto fun = func_list[i];
		gen_cpp_obj << format("warp_point_%d = (void*)GetProcAddress(hm, \"%s\");\n", i, fun.c_str());
	}
	gen_cpp_obj << ("}\n");

	gen_cpp_obj.close();



	std::ofstream gen_asm_obj;
	gen_asm_obj.open("warp_gen.asm", std::ios::out | std::ios::trunc);
	for (size_t i = 0; i < func_list.size(); ++i)
	{
		auto fun = func_list[i];
		gen_asm_obj << format("EXTERN warp_point_%d:QWORD;\n", i);
	}
	gen_asm_obj << ".CODE\n";
	for (size_t i = 0; i < func_list.size(); ++i)
	{
		auto fun = func_list[i];
		gen_asm_obj << format("warp_func_%d PROC\njmp warp_point_%d;\nwarp_func_%d ENDP\n", i,i,i);
	}
	gen_asm_obj << "END\n";
	gen_asm_obj.close();
}
