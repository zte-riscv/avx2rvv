#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <string>
#include "sse_impl.h"

struct TestOptions {
    bool help = false;
    bool list_tests = false;
    bool verbose = false;
    bool quiet = false;
    std::string test_name;
    int test_index = -1;
    bool run_all = true;
};

void print_help(const char* program_name) {
    printf("AVX2RVV Test Suite\n");
    printf("Usage: %s [OPTIONS] [TEST_NAME]\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -l, --list              List all available test cases\n");
    printf("  -v, --verbose           Enable verbose output\n");
    printf("  -q, --quiet             Suppress output except for errors\n");
    printf("  -i, --index INDEX       Run test by index number\n");
    printf("  TEST_NAME               Run specific test by name (supports partial matching)\n\n");
    printf("Examples:\n");
    printf("  %s                      # Run all tests\n", program_name);
    printf("  %s mm_add_ps            # Run mm_add_ps test\n", program_name);
    printf("  %s --index 5            # Run test at index 5\n", program_name);
    printf("  %s --list               # List all available tests\n", program_name);
    printf("  %s --verbose add        # Run tests matching 'add' with verbose output\n", program_name);
}

TestOptions parse_arguments(int argc, const char** argv) {
    TestOptions options;
    
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            options.help = true;
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--list") == 0) {
            options.list_tests = true;
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            options.verbose = true;
        } else if (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0) {
            options.quiet = true;
        } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--index") == 0) {
            if (i + 1 < argc) {
                options.test_index = atoi(argv[++i]);
                options.run_all = false;
            } else {
                fprintf(stderr, "Error: --index requires a number\n");
                exit(1);
            }
        } else if (arg[0] != '-') {
            
            options.test_name = arg;
            options.run_all = false;
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", arg);
            fprintf(stderr, "Use --help for usage information\n");
            exit(1);
        }
    }
    
    return options;
}

void list_tests() {
    printf("Available test cases:\n");
    printf("Index | Test Name\n");
    printf("------|----------\n");
    
    for (uint32_t i = 0; i < SSE2RVV::it_last; i++) {
        printf("%5u | %s\n", i, SSE2RVV::instruction_string[i]);
    }
    printf("\nTotal: %u test cases\n", SSE2RVV::it_last);
}

std::vector<uint32_t> find_tests(const std::string& pattern) {
    std::vector<uint32_t> matches;
    std::string lower_pattern = pattern;
    std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), ::tolower);
    
    for (uint32_t i = 0; i < SSE2RVV::it_last; i++) {
        std::string test_name = SSE2RVV::instruction_string[i];
        std::transform(test_name.begin(), test_name.end(), test_name.begin(), ::tolower);
        
        if (test_name.find(lower_pattern) != std::string::npos) {
            matches.push_back(i);
        }
    }
    
    return matches;
}

bool validate_test_index(int index) {
    return index >= 0 && index < static_cast<int>(SSE2RVV::it_last);
}

SSE2RVV::result_t run_single_test(SSE2RVV::SSE2RVV_TEST* test, uint32_t test_index, bool verbose) {
    SSE2RVV::INSTRUCTION_TEST it = SSE2RVV::INSTRUCTION_TEST(test_index);
    SSE2RVV::result_t ret = test->run_test(it);
    
    if (verbose) {
        printf("Running test %u: %s... ", test_index, SSE2RVV::instruction_string[it]);
        fflush(stdout);
    }
    
    return ret;
}

void print_test_result(uint32_t test_index, SSE2RVV::result_t result, bool verbose) {
    const char* test_name = SSE2RVV::instruction_string[test_index];
    
    if (verbose) {
        switch (result) {
            case SSE2RVV::TEST_SUCCESS:
                printf("PASSED\n");
                break;
            case SSE2RVV::TEST_FAIL:
                printf("FAILED\n");
                break;
            case SSE2RVV::TEST_UNIMPL:
                printf("SKIPPED\n");
                break;
        }
    } else {
        switch (result) {
            case SSE2RVV::TEST_SUCCESS:
                printf("Test %-30s passed\n", test_name);
                break;
            case SSE2RVV::TEST_FAIL:
                printf("Test %-30s failed\n", test_name);
                break;
            case SSE2RVV::TEST_UNIMPL:
                printf("Test %-30s skipped\n", test_name);
                break;
        }
    }
}

void print_test_summary(uint32_t pass_count, uint32_t failed_count, uint32_t ignore_count, bool quiet) {
    if (quiet) {
        if (failed_count > 0) {
            printf("Failed: %u\n", failed_count);
        }
    } else {
        printf("\nSSE2RVV_TEST Complete!\n");
        printf("Passed:  %u\n", pass_count);
        printf("Failed:  %u\n", failed_count);
        printf("Ignored: %u\n", ignore_count);
        
        uint32_t total = pass_count + failed_count + ignore_count;
        if (total > 0) {
            printf("Coverage rate: %.2f%%\n", (float)pass_count / total * 100);
        }
    }
}

int main(int argc, const char** argv) {
    TestOptions options = parse_arguments(argc, argv);
    
    if (options.help) {
        print_help(argv[0]);
        return 0;
    }
    
    if (options.list_tests) {
        list_tests();
        return 0;
    }
    
    SSE2RVV::SSE2RVV_TEST* test = SSE2RVV::SSE2RVV_TEST::create();
    if (!test) {
        fprintf(stderr, "Error: Failed to create test instance\n");
        return -1;
    }
    
    uint32_t pass_count = 0;
    uint32_t failed_count = 0;
    uint32_t ignore_count = 0;
    std::vector<uint32_t> tests_to_run;
    
    if (options.run_all) {
        for (uint32_t i = 0; i < SSE2RVV::it_last; i++) {
            tests_to_run.push_back(i);
        }
    } else if (options.test_index >= 0) {
        if (!validate_test_index(options.test_index)) {
            fprintf(stderr, "Error: Test index %d is out of range (0-%u)\n", 
                    options.test_index, SSE2RVV::it_last - 1);
            test->release();
            return -2;
        }
        tests_to_run.push_back(options.test_index);
    } else if (!options.test_name.empty()) {
        tests_to_run = find_tests(options.test_name);
        if (tests_to_run.empty()) {
            fprintf(stderr, "Error: No test found matching '%s'\n", options.test_name.c_str());
            fprintf(stderr, "Use --list to see all available tests\n");
            test->release();
            return -2;
        }
        
        if (tests_to_run.size() > 1 && !options.verbose) {
            printf("Found %zu matching tests:\n", tests_to_run.size());
            for (uint32_t index : tests_to_run) {
                printf("  %u: %s\n", index, SSE2RVV::instruction_string[index]);
            }
            printf("\n");
        }
    }
    
    for (uint32_t test_index : tests_to_run) {
        SSE2RVV::result_t result = run_single_test(test, test_index, options.verbose);
        print_test_result(test_index, result, options.verbose);
        
        switch (result) {
            case SSE2RVV::TEST_SUCCESS:
                pass_count++;
                break;
            case SSE2RVV::TEST_FAIL:
                failed_count++;
                break;
            case SSE2RVV::TEST_UNIMPL:
                ignore_count++;
                break;
        }
    }
    
    print_test_summary(pass_count, failed_count, ignore_count, options.quiet);
    
    test->release();
    
    return failed_count > 0 ? -1 : 0;
}