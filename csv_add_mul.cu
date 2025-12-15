// 从两个 CSV 中读取 1024 个整数，打包到 BFV 槽位后加密，执行同态加法/乘法，解密结果再写回 CSV。
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

using namespace std;

#include "include/Plaintext.cuh"
#include "include/Ciphertext.cuh"
#include "include/BFVScheme.cuh"
#include "include/BFVcontext.cuh"

namespace {
constexpr size_t kCsvSize = 1024;          // 每个输入/输出 CSV 期望的整数个数
constexpr size_t kDefaultPolyDegree = 32768; // 默认多项式模次数，与仓库示例一致

string trim(const string &s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

vector<int64_t> readCsv(const string &path) {
    ifstream fin(path);
    if (!fin.is_open()) throw runtime_error("无法打开输入文件: " + path);
    vector<int64_t> values;
    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            cell = trim(cell);
            if (cell.empty()) continue;
            values.push_back(stoll(cell));
        }
    }
    if (values.size() < kCsvSize) throw runtime_error("文件 " + path + " 中的数值不足 1024 个");
    if (values.size() > kCsvSize) values.resize(kCsvSize); // 只取前 1024 个
    return values;
}

void writeCsv(const string &path, const vector<uint64_tt> &values) {
    ofstream fout(path);
    if (!fout.is_open()) throw runtime_error("无法写出到文件: " + path);
    for (size_t i = 0; i < values.size(); ++i) {
        fout << values[i];
        if (i + 1 < values.size()) fout << ",";
    }
    fout << "\n";
}

// 将 host 整数数据填入长度为 slots 的 uint64_tt 向量，超出部分补 0，并按明文模数取模。
void packIntsToSlots(const vector<int64_t> &src, vector<uint64_tt> &dst, uint64_tt plain_modulus) {
    fill(dst.begin(), dst.end(), 0);
    const size_t count = min(src.size(), dst.size());
    for (size_t i = 0; i < count; ++i) {
        int64_t v = src[i];
        // 映射到 [0, plain_modulus) 区间
        int64_t modded = v % static_cast<int64_t>(plain_modulus);
        if (modded < 0) modded += static_cast<int64_t>(plain_modulus);
        dst[i] = static_cast<uint64_tt>(modded);
    }
}

// 解密 + decode，返回前 1024 个槽位的整数
vector<uint64_tt> decryptAndDecode(BFVContext &context, BFVScheme &scheme, SecretKey &sk,
                                   Ciphertext &ct, int N, int L, int slots) {
    Plaintext plain_dec(N, L, L, slots);
    scheme.decryptMsg(plain_dec, sk, ct);

    uint64_tt *device_dec = nullptr;
    cudaMalloc(&device_dec, sizeof(uint64_tt) * slots);
    context.decode(plain_dec, device_dec);

    vector<uint64_tt> host_dec(slots);
    cudaMemcpy(host_dec.data(), device_dec, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToHost);
    cudaFree(device_dec);

    vector<uint64_tt> first(kCsvSize, 0);
    const size_t copyCount = min(kCsvSize, static_cast<size_t>(slots));
    for (size_t i = 0; i < copyCount; ++i) first[i] = host_dec[i];
    return first;
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cout << "用法: " << argv[0]
             << " <csv1> <csv2> <out_add_csv> <out_mul_csv> [poly_modulus_degree]" << endl;
        return 1;
    }

    const string in_csv1 = argv[1];
    const string in_csv2 = argv[2];
    const string out_add_csv = argv[3];
    const string out_mul_csv = argv[4];
    const size_t poly_modulus_degree = (argc >= 6) ? static_cast<size_t>(stoull(argv[5])) : kDefaultPolyDegree;

    try {
        cout << "读取 CSV..." << endl;
        const vector<int64_t> data1 = readCsv(in_csv1);
        const vector<int64_t> data2 = readCsv(in_csv2);

        cout << "初始化 BFV 上下文..." << endl;
        BFVContext context(poly_modulus_degree);
        BFVScheme scheme(context);
        SecretKey sk(context);
        scheme.mallocMemory();
        scheme.addEncKey(sk);
        scheme.addMultKey_23(sk);
        scheme.addLeftRotKey_23(sk, 1);

        const int N = context.N;
        const int L = context.L;
        const int slots = context.slots;
        const uint64_tt plain_mod = context.plain_modulus;
        cout << "总槽位数: " << slots << ", 仅使用前 " << kCsvSize << " 个槽位写入 CSV 数据" << endl;

        vector<uint64_tt> mes1(slots);
        vector<uint64_tt> mes2(slots);
        packIntsToSlots(data1, mes1, plain_mod);
        packIntsToSlots(data2, mes2, plain_mod);

        uint64_tt *d_msg1 = nullptr;
        uint64_tt *d_msg2 = nullptr;
        cudaMalloc(&d_msg1, sizeof(uint64_tt) * slots);
        cudaMalloc(&d_msg2, sizeof(uint64_tt) * slots);
        cudaMemcpy(d_msg1, mes1.data(), sizeof(uint64_tt) * slots, cudaMemcpyHostToDevice);
        cudaMemcpy(d_msg2, mes2.data(), sizeof(uint64_tt) * slots, cudaMemcpyHostToDevice);

        Plaintext plain1(N, L, L, slots);
        Plaintext plain2(N, L, L, slots);
        Ciphertext c1(N, L, L, slots);
        Ciphertext c2(N, L, L, slots);

        cout << "编码并加密..." << endl;
        context.encode(d_msg1, plain1);
        context.encode(d_msg2, plain2);
        scheme.encryptMsg(c1, plain1);
        scheme.encryptMsg(c2, plain2);

        // 密文加法
        cout << "执行密文加法..." << endl;
        Ciphertext c_add = c1;
        scheme.addAndEqual(c_add, c2);

        // 密文乘法
        cout << "执行密文乘法..." << endl;
        Ciphertext c_mul = c1;
        scheme.multAndEqual_23(c_mul, c2);

        cout << "解密/解码并写出 CSV..." << endl;
        const vector<uint64_tt> add_result = decryptAndDecode(context, scheme, sk, c_add, N, L, slots);
        const vector<uint64_tt> mul_result = decryptAndDecode(context, scheme, sk, c_mul, N, L, slots);
        writeCsv(out_add_csv, add_result);
        writeCsv(out_mul_csv, mul_result);

        cudaFree(d_msg1);
        cudaFree(d_msg2);

        cout << "完成。结果已写入: " << out_add_csv << " 和 " << out_mul_csv << endl;
    } catch (const exception &ex) {
        cerr << "运行时出错: " << ex.what() << endl;
        return 1;
    }

    return 0;
}
