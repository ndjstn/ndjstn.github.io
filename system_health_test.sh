#!/bin/bash

# Define output file
output_file="system_health_report_$(date +%Y%m%d%H%M%S).txt"

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null
}

# Function to install a missing command
install_command() {
    echo "Installing $1..."
    sudo apt-get update
    sudo apt-get install -y $1
}

# Start the report
echo "Starting System Health Check..." | tee $output_file
echo "Report generated on $(date)" | tee -a $output_file
echo "-------------------------------------------------" | tee -a $output_file

# Checking and installing dependencies
DEPENDENCIES=(dmidecode lscpu sysbench smartctl ifconfig speedtest-cli sensors upower)

for cmd in "${DEPENDENCIES[@]}"; do
    if ! command_exists $cmd; then
        install_command $cmd
    fi
done

# System Information
echo "Gathering System Information..." | tee -a $output_file
echo "System Manufacturer, Model, and Serial Number:" | tee -a $output_file
sudo dmidecode -t system | grep 'Manufacturer\|Product Name\|Serial Number' | tee -a $output_file



# Start the report
echo "Starting System Health Check..." | tee $output_file
echo "Report generated on $(date)" | tee -a $output_file
echo "-------------------------------------------------" | tee -a $output_file

# System Information
echo "Gathering System Information..." | tee -a $output_file
echo "System Manufacturer, Model, and Serial Number:" | tee -a $output_file
sudo dmidecode -t system | grep 'Manufacturer\|Product Name\|Serial Number' | tee -a $output_file

echo "CPU Information:" | tee -a $output_file
lscpu | grep 'Model name\|CPU(s)\|Thread(s) per core\|Core(s) per socket' | tee -a $output_file

echo "Memory Information:" | tee -a $output_file
free -h | tee -a $output_file

# CPU Performance Test
echo "Performing CPU Performance Test..." | tee -a $output_file
sysbench --test=cpu --cpu-max-prime=20000 run | tee -a $output_file

# Memory Test
echo "Performing Memory Test (Quick Check)..." | tee -a $output_file
sudo dmidecode -t memory | grep 'Size:' | tee -a $output_file

# Disk Health Check
echo "Checking Disk Health..." | tee -a $output_file
df -h | tee -a $output_file
echo "SMART Status:" | tee -a $output_file
sudo smartctl -a /dev/sda | tee -a $output_file

# Network Interface Check
echo "Checking Network Interfaces..." | tee -a $output_file
ifconfig -a | tee -a $output_file
echo "Performing Network Speed Test..." | tee -a $output_file
speedtest-cli | tee -a $output_file

# Temperature Monitoring
echo "Checking System Temperatures..." | tee -a $output_file
sensors | tee -a $output_file

# Battery Health Check (if applicable)
echo "Checking Battery Health..." | tee -a $output_file
upower -i /org/freedesktop/UPower/devices/battery_BAT0 | tee -a $output_file

echo "System Health Check Complete." | tee -a $output_file

echo "Report saved to $output_file"
