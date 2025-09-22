#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cctype>
#include "sse_impl.h"
#include "avx_impl.h"

enum class TestSuite {
    SSE,
    AVX,
    ALL
};

using TestResult = SSE2RVV::result_t;
constexpr TestResult TEST_SUCCESS = SSE2RVV::TEST_SUCCESS;
constexpr TestResult TEST_FAIL = SSE2RVV::TEST_FAIL;
constexpr TestResult TEST_UNIMPL = SSE2RVV::TEST_UNIMPL;

struct TestOptions {
    bool show_help = false;
    bool list_tests = false;
    bool verbose_output = false;
    bool quiet_output = false;
    std::string target_test_name;
    int target_test_index = -1;
    bool run_all_tests = true;
    TestSuite target_suite = TestSuite::ALL;
};

static void to_lower_inplace(std::string& str) {
    std::transform(str.begin(), str.end(), str.begin(), 
                  [](unsigned char c) { return std::tolower(c); });
}

static bool str_to_test_suite(const std::string& suite_str, TestSuite& out_suite) {
    std::string lower_str = suite_str;
    to_lower_inplace(lower_str);
    
    if (lower_str == "sse") {
        out_suite = TestSuite::SSE;
        return true;
    } else if (lower_str == "avx") {
        out_suite = TestSuite::AVX;
        return true;
    } else if (lower_str == "all") {
        out_suite = TestSuite::ALL;
        return true;
    } else {
        return false;
    }
}

static const char* get_suite_name(TestSuite suite) {
    switch (suite) {
        case TestSuite::SSE: return "SSE";
        case TestSuite::AVX: return "AVX";
        case TestSuite::ALL: return "ALL (SSE → AVX)";
        default: return "Unknown";
    }
}

static uint32_t get_single_suite_test_count(TestSuite suite) {
    switch (suite) {
        case TestSuite::SSE: return SSE2RVV::it_last;
        case TestSuite::AVX: return AVX2RVV::it_last;
        default: return 0;
    }
}

static const char* get_single_suite_test_name(TestSuite suite, uint32_t test_index) {
    switch (suite) {
        case TestSuite::SSE: return SSE2RVV::instruction_string[test_index];
        case TestSuite::AVX: return AVX2RVV::instruction_string[test_index];
        default: return "Unknown Test";
    }
}

static std::vector<TestSuite> get_all_suites() {
    return {TestSuite::SSE, TestSuite::AVX};
}

static void print_help(const char* program_name) {
    printf("AVX2RVV Test Suite\n");
    printf("Usage: %s [OPTIONS] [TEST_NAME]\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help                 Show this help message\n");
    printf("  -l, --list                 List all available test cases\n");
    printf("  -v, --verbose              Enable verbose output (per-test progress)\n");
    printf("  -q, --quiet                Suppress output except for errors\n");
    printf("  -i, --index INDEX          Run test by index number (per suite)\n");
    printf("  -s, --suite sse|avx|all    Select test suite (default: all → run SSE first, then AVX)\n");
    printf("  TEST_NAME                  Run specific test by name (supports partial matching)\n\n");
    printf("Examples:\n");
    printf("  %s                         # Run all SSE tests, then all AVX tests\n", program_name);
    printf("  %s --suite avx             # Run only AVX tests\n", program_name);
    printf("  %s --suite all mm_add       # Run 'mm_add' tests in SSE, then AVX\n", program_name);
    printf("  %s --suite sse --index 5    # Run SSE test at index 5\n", program_name);
}

static TestOptions parse_arguments(int argc, const char** argv) {
    TestOptions options;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "-i") == 0 || strcmp(arg, "--index") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --index requires a numeric argument (e.g., --index 5)\n");
                exit(EXIT_FAILURE);
            }
            const char* index_str = argv[++i];
            if (strspn(index_str, "0123456789") != strlen(index_str)) {
                fprintf(stderr, "Error: --index argument must be a non-negative integer (got '%s')\n", index_str);
                exit(EXIT_FAILURE);
            }
            options.target_test_index = atoi(index_str);
            options.run_all_tests = false;
        } 
        else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--suite") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --suite requires a value (sse|avx|all)\n");
                exit(EXIT_FAILURE);
            }
            const char* suite_str = argv[++i];
            if (!str_to_test_suite(suite_str, options.target_suite)) {
                fprintf(stderr, "Error: --suite must be 'sse', 'avx', or 'all' (got '%s')\n", suite_str);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            options.show_help = true;
        } 
        else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--list") == 0) {
            options.list_tests = true;
        } 
        else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            options.verbose_output = true;
        } 
        else if (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0) {
            options.quiet_output = true;
        }
        else if (arg[0] != '-') {
            options.target_test_name = arg;
            options.run_all_tests = false;
        }
        else {
            fprintf(stderr, "Error: Unknown option '%s'\n", arg);
            fprintf(stderr, "Use '%s --help' for usage information\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return options;
}

static void list_single_suite_tests(TestSuite suite) {
    const uint32_t test_count = get_single_suite_test_count(suite);
    if (test_count == 0) {
        fprintf(stderr, "Warning: No tests available for %s suite\n", get_suite_name(suite));
        return;
    }

    printf("\n%s Suite Test Cases:\n", get_suite_name(suite));
    printf("Index | Test Name\n");
    printf("------|------------------------------\n");

    for (uint32_t i = 0; i < test_count; ++i) {
        printf("%5u | %s\n", i, get_single_suite_test_name(suite, i));
    }

    printf("Total %s tests: %u\n", get_suite_name(suite), test_count);
}

static void list_all_suites_tests() {
    printf("Listing all test cases (priority: SSE → AVX)\n");
    for (const auto& suite : get_all_suites()) {
        list_single_suite_tests(suite);
    }
}

static void list_tests(TestSuite suite) {
    if (suite == TestSuite::ALL) {
        list_all_suites_tests();
    } else {
        list_single_suite_tests(suite);
    }
}

static bool is_valid_single_suite_index(TestSuite suite, int test_index) {
    if (test_index < 0) return false;
    const uint32_t max_index = get_single_suite_test_count(suite) - 1;
    return static_cast<uint32_t>(test_index) <= max_index;
}

static std::vector<uint32_t> find_single_suite_matching_tests(TestSuite suite, const std::string& name_pattern) {
    std::vector<uint32_t> matches;
    const uint32_t test_count = get_single_suite_test_count(suite);
    if (test_count == 0) return matches;

    std::string lower_pattern = name_pattern;
    to_lower_inplace(lower_pattern);

    for (uint32_t i = 0; i < test_count; ++i) {
        std::string test_name = get_single_suite_test_name(suite, i);
        to_lower_inplace(test_name);
        if (test_name.find(lower_pattern) != std::string::npos) {
            matches.push_back(i);
        }
    }

    return matches;
}

// static void find_all_suites_matching_tests(const std::string& name_pattern, 
//                                           std::vector<TestSuite>& out_suites,
//                                           std::vector<std::vector<uint32_t>>& out_test_indices) {
//     out_suites.clear();
//     out_test_indices.clear();

//     for (const auto& suite : get_all_suites()) {
//         const auto matches = find_single_suite_matching_tests(suite, name_pattern);
//         if (!matches.empty()) {
//             out_suites.push_back(suite);
//             out_test_indices.push_back(matches);
//         }
//     }
// }

template <typename TestType>
static bool create_test_instance(TestType*& out_instance, const char* suite_name) {
    out_instance = TestType::create();
    if (!out_instance) {
        fprintf(stderr, "Error: Failed to create %s test instance\n", suite_name);
        return false;
    }
    return true;
}

template <typename TestType>
static void release_test_instance(TestType*& instance) {
    if (instance) {
        instance->release();
        instance = nullptr;
    }
}

static TestResult run_sse_test(SSE2RVV::SSE2RVV_TEST* test, uint32_t test_index, bool verbose) {
    const char* test_name = get_single_suite_test_name(TestSuite::SSE, test_index);
    if (verbose) {
        printf("[SSE] Running test %u: %s... ", test_index, test_name);
        fflush(stdout);
    }

    const auto test_type = static_cast<SSE2RVV::INSTRUCTION_TEST>(test_index);
    return test->run_test(test_type);
}

static TestResult run_avx_test(AVX2RVV::AVX2RVV_TEST* test, uint32_t test_index, bool verbose) {
    const char* test_name = get_single_suite_test_name(TestSuite::AVX, test_index);
    if (verbose) {
        printf("[AVX] Running test %u: %s... ", test_index, test_name);
        fflush(stdout);
    }

    const auto test_type = static_cast<AVX2RVV::INSTRUCTION_TEST>(test_index);
    return test->run_test(test_type);
}

static void print_test_result(TestSuite suite, uint32_t test_index, TestResult result, bool verbose) {
    const char* test_name = get_single_suite_test_name(suite, test_index);
    const char* result_str = nullptr;

    switch (result) {
        case TEST_SUCCESS: result_str = "PASSED"; break;
        case TEST_FAIL:    result_str = "FAILED"; break;
        case TEST_UNIMPL:  result_str = "SKIPPED"; break;
        default:           result_str = "UNKNOWN";
    }

    if (verbose) {
        printf("%s\n", result_str);
    } else {
        printf("[%s] Test %-30s %s\n", get_suite_name(suite), test_name, result_str);
    }
}

static bool run_single_suite_tests(TestSuite suite,
                                   const std::vector<uint32_t>& test_indices,
                                   bool verbose,
                                   uint32_t& out_pass,
                                   uint32_t& out_fail,
                                   uint32_t& out_skip) {
    out_pass = 0;
    out_fail = 0;
    out_skip = 0;

    SSE2RVV::SSE2RVV_TEST* sse_test = nullptr;
    AVX2RVV::AVX2RVV_TEST* avx_test = nullptr;

    if (suite == TestSuite::SSE) {
        if (!create_test_instance(sse_test, "SSE")) return false;
    } else if (suite == TestSuite::AVX) {
        if (!create_test_instance(avx_test, "AVX")) return false;
    } else {
        fprintf(stderr, "Error: Unsupported suite for single run\n");
        return false;
    }

    for (const auto& test_idx : test_indices) {
        TestResult result;
        if (suite == TestSuite::SSE) {
            result = run_sse_test(sse_test, test_idx, verbose);
        } else {
            result = run_avx_test(avx_test, test_idx, verbose);
        }

        print_test_result(suite, test_idx, result, verbose);
        switch (result) {
            case TEST_SUCCESS: out_pass++; break;
            case TEST_FAIL:    out_fail++; break;
            case TEST_UNIMPL:  out_skip++; break;
            default: break;
        }
    }

    release_test_instance(sse_test);
    release_test_instance(avx_test);
    return true;
}

static bool run_all_suites_tests(bool run_all,
                                 int target_index,
                                 const std::string& target_name,
                                 bool verbose,
                                 uint32_t& out_total_pass,
                                 uint32_t& out_total_fail,
                                 uint32_t& out_total_skip) {
    out_total_pass = 0;
    out_total_fail = 0;
    out_total_skip = 0;

    for (const auto& suite : get_all_suites()) {
        std::vector<uint32_t> test_indices;
        const char* suite_name = get_suite_name(suite);
        const uint32_t test_count = get_single_suite_test_count(suite);

        if (run_all) {
            for (uint32_t i = 0; i < test_count; ++i) {
                test_indices.push_back(i);
            }
            printf("\n=== Starting %s suite (total %u tests) ===\n", suite_name, test_count);
        }
        else if (target_index >= 0) {
            if (!is_valid_single_suite_index(suite, target_index)) {
                fprintf(stderr, "Warning: Index %d out of range for %s suite (0-%u), skipping\n",
                        target_index, suite_name, test_count - 1);
                continue;
            }
            test_indices.push_back(static_cast<uint32_t>(target_index));
            printf("\n=== Running %s suite test index %d ===\n", suite_name, target_index);
        }
        else if (!target_name.empty()) {
            test_indices = find_single_suite_matching_tests(suite, target_name);
            if (test_indices.empty()) {
                printf("\nNo matching tests for pattern '%s' in %s suite\n", target_name.c_str(), suite_name);
                continue;
            }
            printf("\n=== Found %zu matching tests in %s suite ===\n", test_indices.size(), suite_name);
            if (!verbose) {
                for (uint32_t idx : test_indices) {
                    printf("  %u: %s\n", idx, get_single_suite_test_name(suite, idx));
                }
                printf("\n");
            }
        }

        if (test_indices.empty()) continue;

        uint32_t pass, fail, skip;
        if (!run_single_suite_tests(suite, test_indices, verbose, pass, fail, skip)) {
            fprintf(stderr, "Error: Failed to run %s suite tests\n", suite_name);
            return false;
        }

        out_total_pass += pass;
        out_total_fail += fail;
        out_total_skip += skip;
    }

    return true;
}

static void print_test_summary(uint32_t pass_count, uint32_t fail_count, uint32_t skip_count, bool quiet, TestSuite suite) {
    const uint32_t total_count = pass_count + fail_count + skip_count;

    if (quiet) {
        if (fail_count > 0) {
            printf("Failed: %u\n", fail_count);
        }
        return;
    }

    printf("\n=== %s Test Suite Summary ===\n", get_suite_name(suite));
    printf("Total tests:   %u\n", total_count);
    printf("Passed:        %u\n", pass_count);
    printf("Failed:        %u\n", fail_count);
    printf("Skipped:       %u\n", skip_count);

    if (total_count > 0) {
        const double coverage = static_cast<double>(pass_count) / total_count * 100.0;
        printf("Coverage rate: %.2f%%\n", coverage);
    }
}

int main(int argc, const char** argv) {
    TestOptions options = parse_arguments(argc, argv);

    if (options.show_help) {
        print_help(argv[0]);
        return 0;
    }

    if (options.list_tests) {
        list_tests(options.target_suite);
        return 0;
    }

    uint32_t pass_count = 0;
    uint32_t fail_count = 0;
    uint32_t skip_count = 0;
    bool run_success = false;

    if (options.target_suite == TestSuite::ALL) {
        run_success = run_all_suites_tests(
            options.run_all_tests,
            options.target_test_index,
            options.target_test_name,
            options.verbose_output,
            pass_count,
            fail_count,
            skip_count
        );
    } else {
        std::vector<uint32_t> test_indices;
        const char* suite_name = get_suite_name(options.target_suite);
        const uint32_t test_count = get_single_suite_test_count(options.target_suite);

        if (options.run_all_tests) {
            for (uint32_t i = 0; i < test_count; ++i) {
                test_indices.push_back(i);
            }
            printf("=== Starting %s suite (total %u tests) ===\n", suite_name, test_count);
        }
        else if (options.target_test_index >= 0) {
            if (!is_valid_single_suite_index(options.target_suite, options.target_test_index)) {
                fprintf(stderr, "Error: Test index %d is out of range for %s suite (0-%u)\n",
                        options.target_test_index, suite_name, test_count - 1);
                return EXIT_FAILURE;
            }
            test_indices.push_back(static_cast<uint32_t>(options.target_test_index));
        }
        else if (!options.target_test_name.empty()) {
            test_indices = find_single_suite_matching_tests(options.target_suite, options.target_test_name);
            if (test_indices.empty()) {
                fprintf(stderr, "Error: No test found matching '%s' in %s suite\n",
                        options.target_test_name.c_str(), suite_name);
                fprintf(stderr, "Use --list to see available tests\n");
                return EXIT_FAILURE;
            }
            printf("Found %zu matching tests in %s suite:\n", test_indices.size(), suite_name);
            if (!options.verbose_output) {
                for (uint32_t idx : test_indices) {
                    printf("  %u: %s\n", idx, get_single_suite_test_name(options.target_suite, idx));
                }
                printf("\n");
            }
        }

        run_success = run_single_suite_tests(
            options.target_suite,
            test_indices,
            options.verbose_output,
            pass_count,
            fail_count,
            skip_count
        );
    }

    if (!run_success) {
        fprintf(stderr, "Error: Test execution failed\n");
        return EXIT_FAILURE;
    }

    print_test_summary(pass_count, fail_count, skip_count, options.quiet_output, options.target_suite);

    return (fail_count > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
