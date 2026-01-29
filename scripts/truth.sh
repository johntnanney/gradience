#!/bin/bash
#
# Truth Commands: Define what "sanity" means for Gradience
# 
# These are the acceptance gate commands that define if the codebase is coherent.
# If these pass, the codebase is sane. If a dev script breaks, who cares.
#
# Usage: ./scripts/truth.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}🔍 TRUTH COMMANDS: Acceptance Gate${NC}"
echo "=================================="
echo "These commands define if the codebase is coherent."
echo "If these pass, we are sane. If dev scripts break, who cares."
echo

# Truth Command A: Local Sanity (CI-grade tests)
echo -e "${YELLOW}Truth Command A: Local Sanity${NC}"
echo "Command: pytest -q"
echo "Purpose: All CI-grade tests pass"
echo "Scope:   tests/ directory only (dev/ ignored by pytest.ini)"
echo

if pytest -q > /tmp/truth_pytest.log 2>&1; then
    echo -e "${GREEN}✅ Truth Command A: PASS${NC}"
    echo "   $(grep -o '[0-9]\+ passed\|[0-9]\+ skipped' /tmp/truth_pytest.log | tr '\n' ', ' | sed 's/, $//')"
else
    echo -e "${RED}❌ Truth Command A: FAIL${NC}"
    echo "   CI-grade tests are failing. Check /tmp/truth_pytest.log"
    exit 1
fi

echo

# Truth Command B: CLI Sanity (Core interfaces work)
echo -e "${YELLOW}Truth Command B: CLI Sanity${NC}"
echo "Commands: Core CLI interfaces work"
echo "Purpose:  Entry points are functional"
echo

# B.1: Main CLI help
if python3 -m gradience --help > /tmp/truth_main_cli.log 2>&1; then
    echo -e "${GREEN}✅ python3 -m gradience --help${NC}"
else
    echo -e "${RED}❌ python3 -m gradience --help${NC}"
    echo "   Main CLI broken. Check /tmp/truth_main_cli.log"
    exit 1
fi

# B.2: Audit CLI help  
if python3 -m gradience audit --help > /tmp/truth_audit_cli.log 2>&1; then
    echo -e "${GREEN}✅ python3 -m gradience audit --help${NC}"
else
    echo -e "${RED}❌ python3 -m gradience audit --help${NC}"
    echo "   Audit CLI broken. Check /tmp/truth_audit_cli.log"
    exit 1
fi

# B.3: Bench CLI help
if python3 -m gradience.bench.run_bench --help > /tmp/truth_bench_cli.log 2>&1; then
    echo -e "${GREEN}✅ python3 -m gradience.bench.run_bench --help${NC}"
else
    echo -e "${RED}❌ python3 -m gradience.bench.run_bench --help${NC}"
    echo "   Bench CLI broken. Check /tmp/truth_bench_cli.log"
    exit 1
fi

echo
echo -e "${BOLD}${GREEN}🎉 ALL TRUTH COMMANDS PASS${NC}"
echo "The codebase is coherent and sane."
echo
echo -e "${BLUE}ℹ️  These are your acceptance gates:${NC}"
echo "• Before any commit: ./scripts/truth.sh"  
echo "• Before any release: ./scripts/truth.sh"
echo "• CI failure investigation: ./scripts/truth.sh"
echo
echo -e "${BLUE}💡 Dev script failures don't matter if truth commands pass.${NC}"