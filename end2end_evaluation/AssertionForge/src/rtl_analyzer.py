# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python3



from utils_LLM import count_prompt_tokens
import os
import sys
import re
import argparse
from typing import Dict, List, Set, Tuple, Any
import networkx as nx
from pyverilog.vparser.parser import parse
import pyverilog.vparser.ast as ast
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.walker import VerilogDataflowWalker


# from saver import saver
# print = saver.log_info # would print to log.txt in the log folder


class RTLAnalyzer:
    def __init__(self, design_dir: str, verbose: bool = False):
        self.design_dir = design_dir
        self.verbose = verbose
        self.file_info = {}  # Information about each file
        self.module_info = {}  # Information about each module
        self.control_flow = {}  # Control flow relationships
        self.fsm_info = {}  # FSM information
        self.signal_type_info = {}  # Signal type information
        self.primary_signals = set()  # I/O ports
        self.data_flow_graph = nx.DiGraph()  # Data flow graph
        self.verification_suggestions = []  # Generic verification suggestions

    def analyze_design(self):
        """Main method to analyze all Verilog files in the design directory"""
        print(f"\nAnalyzing RTL design in directory: {self.design_dir}")

        # Get sorted files in dependency order (included files first)
        verilog_files = process_files_in_order(self.design_dir)

        print(f"Found {len(verilog_files)} Verilog file(s) to analyze: {verilog_files}")

        # First pass: collect content from all files
        combined_content = ""
        for file_path in verilog_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Remove include directives to avoid duplicates
                    content = re.sub(r'`include\s+"[^"]+"\s*', '', content)
                    combined_content += f"\n// File: {os.path.basename(file_path)}\n"
                    combined_content += content
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

        num_tokens_RTL = count_prompt_tokens(combined_content)
        print(f'RTL tokens={num_tokens_RTL}')
        # exit(-1)

        # Create a combined file for analysis
        combined_file = os.path.join(self.design_dir, "_combined_rtl.v")
        try:
            with open(combined_file, 'w') as f:
                f.write(combined_content)
            print(f"Created combined RTL file for analysis: {combined_file}")
        except Exception as e:
            print(f"Error creating combined file: {str(e)}")
            combined_file = None

        self.combined_content = combined_content

        # Try both approaches: combined file and individual files
        if combined_file:
            print("\nAttempting to process combined file...")
            self._process_file(combined_file)

        # If the combined approach didn't yield enough results, try individual files
        if len(self.module_info) == 0:
            print("\nFalling back to processing individual files...")
            # Process each file individually
            for file_path in verilog_files:
                if file_path != combined_file:  # Skip the combined file
                    self._process_file(file_path)

        # Extract port information directly if none found
        if all(
            len(module_data.get('ports', {})) == 0
            for module_data in self.module_info.values()
        ):
            self._extract_port_info_direct()

        # Find module instances directly from files if none found through AST
        if all(
            len(module_data.get('instances', {})) == 0
            for module_data in self.module_info.values()
        ):
            self._find_module_instances_direct()

        # Build data flow graph across modules
        self._build_cross_module_dataflow()

        # Identify FSMs
        self._identify_fsms()

        # Analyze primary signal relationships
        self._analyze_primary_signal_relationships()

        # Print summary
        self._print_summary()

        # Enhanced analysis
        self._enhanced_signal_analysis()

        print("\n" + "=" * 80)
        print("DETAILED RTL ANALYSIS")
        print("=" * 80)

        # For each module, extract detailed information
        for module_name, module_data in self.module_info.items():
            # Find the file containing this module
            module_file = None
            for file_name, file_data in self.file_info.items():
                if module_name in file_data.get('modules', []):
                    module_file = file_data['path']
                    break

            if module_file:
                # Extract control flow structures
                self._extract_control_flow(module_name, module_file)

                # Extract assignment relationships
                self._extract_assignments(module_name, module_file)

                # Extract detailed signal attributes
                self._extract_signal_attributes(module_name, module_file)

        # Print expanded summary including the new information
        self._print_expanded_summary()

        # Clean up the combined file
        if combined_file and os.path.exists(combined_file):
            try:
                os.remove(combined_file)
            except:
                pass

    def _create_simplified_sockit_module(self, file_path):
        """Create a simplified version of sockit_owm that PyVerilog can parse"""
        simplified_content = """
    module sockit_owm (
    input            clk,
    input            rst,
    input            bus_ren,
    input            bus_wen,
    input      [1:0] bus_adr,
    input     [31:0] bus_wdt,
    output    [31:0] bus_rdt,
    output           bus_irq,
    output     [3:0] owr_p,
    output     [3:0] owr_e,
    input      [3:0] owr_i
    );
    assign bus_rdt = 32'h0;
    assign bus_irq = 1'b0;
    assign owr_p = 4'h0;
    assign owr_e = 4'h0;
    endmodule
    """
        # Write to new file
        simplified_path = os.path.join(
            os.path.dirname(file_path), "_simplified_sockit.v"
        )
        with open(simplified_path, 'w') as f:
            f.write(simplified_content)

        return simplified_path

    def _process_file(self, file_path: str):
        """Process a single Verilog file"""
        print(f"\nProcessing file: {file_path}")

        try:

            # First, try to extract module definitions directly from the file
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for include directives
            includes = re.findall(r'`include\s+"([^"]+)"', content)
            if includes:
                print(
                    f"  Found {len(includes)} include directive(s): {', '.join(includes)}"
                )

            # If the file contains 'sockit_owm' in its name or content, try the simplified approach
            if 'sockit' in file_path.lower() or 'sockit' in content.lower(): # a bit hakcy for sockit due to pyverilog having some error on sockit
                try:
                    # Create simplified module
                    simplified_path = self._create_simplified_sockit_module(file_path)

                    # Extract module info directly from simplified file
                    module_info = self._extract_module_info_from_simplified(
                        simplified_path
                    )

                    if module_info:
                        # Store file information
                        filename = os.path.basename(file_path)
                        self.file_info[filename] = {
                            'path': file_path,
                            'modules': list(module_info.keys()),
                            'includes': includes,
                        }

                        # Merge module info
                        self.module_info.update(module_info)

                        print(
                            f"  Successfully processed {filename} using simplified model"
                        )
                        print(
                            f"  Found {len(module_info)} module(s): {', '.join(module_info.keys())}"
                        )

                        # Clean up
                        os.remove(simplified_path)
                        return
                except Exception as e:
                    print(f"  Error with simplified model approach: {str(e)}")
                    if os.path.exists(simplified_path):
                        os.remove(simplified_path)
                    # exit(-1)

            # Check for module definitions with a more flexible pattern
            module_pattern = r'module\s+(\w+)\s*(?:\#\s*\([^)]*\))?\s*\('
            modules_in_file = re.findall(module_pattern, content)
            if modules_in_file:
                print(
                    f"  Found module definition(s) in raw file: {', '.join(modules_in_file)}"
                )

            # Now try to parse with Pyverilog
            include_dirs = [os.path.dirname(file_path)]

            # If this file is included by others and doesn't have module definitions, skip parsing
            if not modules_in_file and "`include" not in content:
                print(
                    f"  No module definitions found in {file_path}, skipping Pyverilog parsing"
                )

                # Still record the file for reference
                filename = os.path.basename(file_path)
                self.file_info[filename] = {
                    'path': file_path,
                    'modules': [],
                    'includes': includes,
                }
                return

            # First do the basic include preprocessing
            preprocessed_content = self._preprocess_includes(content, file_path)

            # Create a temporary file with preprocessed content
            temp_dir = os.path.dirname(file_path)
            temp_file = os.path.join(temp_dir, "_temp_" + os.path.basename(file_path))
            with open(temp_file, 'w') as f:
                f.write(preprocessed_content)
            print(f"  Created enhanced preprocessed file for PyVerilog: {temp_file}")

            # Parse the file
            try:
                ast_output, _ = parse([temp_file], preprocess_include=include_dirs)

                # Extract basic module info
                module_info = self._extract_module_info(ast_output)

                if not module_info:
                    print(f"  No modules found in parsed AST for {file_path}")

                    # If we created a temp file but parsing found no modules,
                    # try direct extraction from content
                    if not modules_in_file:
                        print(
                            f"  Attempting direct module extraction from file content"
                        )
                        module_info = self._extract_module_info_from_content(content)

                # Store file information
                filename = os.path.basename(file_path)
                self.file_info[filename] = {
                    'path': file_path,
                    'modules': list(module_info.keys()) if module_info else [],
                    'includes': includes,
                }

                # Merge module info
                if module_info:
                    self.module_info.update(module_info)

                    # Extract data flow information using Pyverilog's dataflow analyzer
                    if len(module_info) > 0:
                        self._extract_dataflow_info(
                            temp_file, include_dirs
                        )  # Use the preprocessed file

                    print(f"  Successfully processed {filename}")
                    print(
                        f"  Found {len(module_info)} module(s): {', '.join(module_info.keys())}"
                    )

            except Exception as e:
                print(f"  Error in Pyverilog parsing: {str(e)}")
                print(f"  Falling back to direct extraction")

                # exit(-1)

                # Try direct extraction if Pyverilog parsing fails
                module_info = self._extract_module_info_from_content(content)

                if module_info:
                    # Store file information
                    filename = os.path.basename(file_path)
                    self.file_info[filename] = {
                        'path': file_path,
                        'modules': list(module_info.keys()),
                        'includes': includes,
                    }

                    # Merge module info
                    self.module_info.update(module_info)

                    print(
                        f"  Successfully extracted {len(module_info)} module(s) directly: {', '.join(module_info.keys())}"
                    )

            # Clean up temporary file if created
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def _preprocess_includes(self, content, file_path):
        """Preprocess include directives by inlining the included files"""
        # This is a simplified preprocessor to handle basic include directives
        dir_path = os.path.dirname(file_path)

        def replace_include(match):
            include_file = match.group(1)
            include_path = os.path.join(dir_path, include_file)

            if os.path.exists(include_path):
                try:
                    with open(include_path, 'r') as f:
                        included_content = f.read()
                    return included_content
                except Exception as e:
                    print(
                        f"  Warning: Could not read included file {include_path}: {str(e)}"
                    )
            else:
                print(f"  Warning: Included file {include_path} not found")

            # Return empty string if include file can't be processed
            return ""

        # Replace all include directives
        processed = re.sub(r'`include\s+"([^"]+)"', replace_include, content)
        return processed

    def _extract_module_info_from_content(self, content):
        """Extract basic module information directly from file content"""
        module_info = {}

        # Find module definitions
        module_pattern = r'module\s+(\w+)\s*\((.*?)\);(.*?)endmodule'
        module_matches = re.findall(module_pattern, content, re.DOTALL)

        for module_name, port_list, module_body in module_matches:
            print(f"  Extracting info for module (direct): {module_name}")

            # Initialize module info
            module_info[module_name] = {
                'ports': {},
                'params': {},
                'signals': {},
                'instances': {},
                'always_blocks': [],
                'assign_stmts': [],
            }

            # Extract ports
            ports = port_list.split(',')
            for port in ports:
                port = port.strip()
                if port:
                    # Simply store the port name for now
                    module_info[module_name]['ports'][port] = {
                        'direction': 'unknown',  # We'd need more parsing to determine direction
                        'width': None,
                    }

            # Extract input/output declarations
            input_pattern = r'input\s+(?:\[(\d+):(\d+)\])?\s*(\w+)'
            output_pattern = r'output\s+(?:\[(\d+):(\d+)\])?\s*(\w+)'

            for msb, lsb, port_name in re.findall(input_pattern, module_body):
                if port_name in module_info[module_name]['ports']:
                    module_info[module_name]['ports'][port_name]['direction'] = 'input'
                    if msb and lsb:
                        module_info[module_name]['ports'][port_name][
                            'width'
                        ] = f"{msb}:{lsb}"

            for msb, lsb, port_name in re.findall(output_pattern, module_body):
                if port_name in module_info[module_name]['ports']:
                    module_info[module_name]['ports'][port_name]['direction'] = 'output'
                    if msb and lsb:
                        module_info[module_name]['ports'][port_name][
                            'width'
                        ] = f"{msb}:{lsb}"

            # Extract always blocks (simplified)
            always_pattern = r'always\s*@\s*\((.*?)\)'
            for senslist in re.findall(always_pattern, module_body):
                module_info[module_name]['always_blocks'].append(
                    {'senslist': self._parse_senslist(senslist)}
                )

        return module_info

    def _parse_senslist(self, senslist):
        """Parse sensitivity list from a string"""
        result = []

        # Split by 'or' if present
        items = senslist.split(' or ')

        # Process each item
        for item in items:
            item = item.strip()

            # Check for posedge/negedge
            if item.startswith('posedge'):
                signal = item.replace('posedge', '').strip()
                result.append({'signal': signal, 'type': 'posedge'})
            elif item.startswith('negedge'):
                signal = item.replace('negedge', '').strip()
                result.append({'signal': signal, 'type': 'negedge'})
            else:
                # Level sensitive
                result.append({'signal': item, 'type': 'level'})

        return result

    def _extract_module_info_from_simplified(self, file_path):
        """Extract module information directly from the simplified file"""
        with open(file_path, 'r') as f:
            content = f.read()

        module_info = {}

        # Extract module name
        module_match = re.search(r'module\s+(\w+)', content)
        if module_match:
            module_name = module_match.group(1)

            # Initialize module info
            module_info[module_name] = {
                'ports': {},
                'params': {},
                'signals': {},
                'instances': {},
                'always_blocks': [],
                'assign_stmts': [],
            }

            # Extract port information
            input_ports = re.findall(
                r'input\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)', content
            )
            output_ports = re.findall(
                r'output\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)', content
            )

            # Process input ports
            for msb, lsb, name in input_ports:
                width = f"{msb}:{lsb}" if msb and lsb else None
                module_info[module_name]['ports'][name] = {
                    'direction': 'input',
                    'width': width,
                }
                # Add to primary signals
                self.primary_signals.add(f"{module_name}.{name}")

            # Process output ports
            for msb, lsb, name in output_ports:
                width = f"{msb}:{lsb}" if msb and lsb else None
                module_info[module_name]['ports'][name] = {
                    'direction': 'output',
                    'width': width,
                }
                # Add to primary signals
                self.primary_signals.add(f"{module_name}.{name}")

            print(
                f"  Extracted {len(module_info[module_name]['ports'])} ports for module {module_name}"
            )

        return module_info

    def _extract_module_info(self, ast_output):
        """Extract information about modules, ports, signals from AST"""
        module_info = {}

        # Debug: Print AST structure
        print(f"  DEBUG: AST type: {type(ast_output)}")

        # The AST structure has Source -> Description -> ModuleDef (nested structure)
        if hasattr(ast_output, 'children') and len(ast_output.children()) > 0:
            description = ast_output.children()[0]
            print(f"  DEBUG: Description type: {type(description)}")

            if hasattr(description, 'definitions'):
                print(f"  DEBUG: Number of definitions: {len(description.definitions)}")
                for i, definition in enumerate(description.definitions):
                    print(f"  DEBUG: Definition {i} type: {type(definition)}")

                    if isinstance(definition, ast.ModuleDef):
                        module_name = definition.name
                        print(f"  Extracting info for module: {module_name}")

                        # Initialize module info
                        module_info[module_name] = {
                            'ports': {},
                            'params': {},
                            'signals': {},
                            'instances': {},
                            'always_blocks': [],
                            'assign_stmts': [],
                        }

                        # Extract port information
                        if definition.portlist:
                            for port in definition.portlist.ports:
                                port_name = port.name
                                port_dir = 'inout'  # Default

                                if isinstance(port, ast.Ioport):
                                    if isinstance(port.first, ast.Input):
                                        port_dir = 'input'
                                    elif isinstance(port.first, ast.Output):
                                        port_dir = 'output'
                                    elif isinstance(port.first, ast.Inout):
                                        port_dir = 'inout'

                                    # Check if port has a width
                                    width = None
                                    if (
                                        hasattr(port.first, 'width')
                                        and port.first.width
                                    ):
                                        width = self._parse_width(port.first.width)

                                    module_info[module_name]['ports'][port_name] = {
                                        'direction': port_dir,
                                        'width': width,
                                    }

                                    # Add to primary signals
                                    self.primary_signals.add(
                                        f"{module_name}.{port_name}"
                                    )

                        # Extract items from the module declaration (signals, instances, etc.)
                        if definition.items:
                            for item in definition.items:
                                # Extract wire and reg declarations
                                if isinstance(
                                    item,
                                    (
                                        ast.Wire,
                                        ast.Reg,
                                        ast.Input,
                                        ast.Output,
                                        ast.Inout,
                                    ),
                                ):
                                    signal_name = item.name
                                    width = None
                                    if hasattr(item, 'width') and item.width:
                                        width = self._parse_width(item.width)

                                    signal_type = type(item).__name__.lower()
                                    module_info[module_name]['signals'][signal_name] = {
                                        'type': signal_type,
                                        'width': width,
                                    }

                                # Extract module instances
                                elif isinstance(item, ast.Instance):
                                    instance_name = item.name
                                    module_type = item.module

                                    module_info[module_name]['instances'][
                                        instance_name
                                    ] = {
                                        'module': module_type,
                                        'connections': {},
                                    }

                                    # Extract port connections
                                    if item.portlist:
                                        for port_conn in item.portlist:
                                            port_name = port_conn.portname
                                            connection = None
                                            if port_conn.argname:
                                                connection = port_conn.argname

                                            module_info[module_name]['instances'][
                                                instance_name
                                            ]['connections'][port_name] = connection

                                # Extract always blocks
                                elif isinstance(item, ast.Always):
                                    senslist = []
                                    if hasattr(item, 'senslist') and item.senslist:
                                        for sens in item.senslist.list:
                                            if isinstance(sens, ast.Sens):
                                                senslist.append(
                                                    {
                                                        'signal': sens.name,
                                                        'type': (
                                                            'posedge'
                                                            if sens.type == 'posedge'
                                                            else (
                                                                'negedge'
                                                                if sens.type
                                                                == 'negedge'
                                                                else 'level'
                                                            )
                                                        ),
                                                    }
                                                )

                                    module_info[module_name]['always_blocks'].append(
                                        {'senslist': senslist}
                                    )

                                # Extract assign statements
                                elif isinstance(item, ast.Assign):
                                    lhs = (
                                        item.left.name
                                        if hasattr(item.left, 'name')
                                        else str(item.left)
                                    )
                                    module_info[module_name]['assign_stmts'].append(
                                        {'lhs': lhs}
                                    )

        return module_info

    def _parse_width(self, width_node) -> str:
        """Parse width information from AST node"""
        if isinstance(width_node, ast.Width):
            msb = width_node.msb
            lsb = width_node.lsb
            if hasattr(msb, 'value') and hasattr(lsb, 'value'):
                return f"{msb.value}:{lsb.value}"
            return "width_expression"
        return None

    def _extract_dataflow_info(self, file_path: str, include_dirs: List[str]):
        """Extract data flow information using Pyverilog's dataflow analyzer"""
        print(f"  Extracting data flow information...")

        try:
            # Top module (will be overridden by actual top module)
            topmodule = None

            # Get the first module name from the file
            for module_name in self.file_info[os.path.basename(file_path)]['modules']:
                if not topmodule:
                    topmodule = module_name

            if not topmodule:
                print(f"  No top module found for data flow analysis in {file_path}")
                return

            # Setup the dataflow analyzer
            analyzer = VerilogDataflowAnalyzer(file_path, topmodule, include_dirs)
            analyzer.generate()

            # Get the dataflow graph
            dfg = analyzer.getCircuit()

            # Setup optimizer
            optimizer = VerilogDataflowOptimizer(dfg)
            optimizer.optimize()

            # Setup walker
            walker = VerilogDataflowWalker(dfg, optimizer)

            # Get all signals defined in the module
            signals = dfg.getSignals()
            print(f"  Found {len(signals)} signals in data flow graph")

            # Extract signal definitions and assignments
            for signal_name, signal_obj in signals.items():
                if topmodule in signal_name:  # Only process signals in this module
                    # Skip temporary signals and constants
                    if signal_name.startswith('_') or signal_name.isdigit():
                        continue

                    # Get the term that defines this signal
                    terms = dfg.getTerms(signal_name)
                    if terms:
                        for term in terms:
                            self._add_to_dataflow_graph(signal_name, term, dfg)

        except Exception as e:
            print(f"  Error in data flow analysis: {str(e)}")

    def _add_to_dataflow_graph(self, signal_name: str, term, dfg):
        """Add information from a term to the data flow graph"""
        # Add the signal node if it doesn't exist
        if signal_name not in self.data_flow_graph:
            self.data_flow_graph.add_node(signal_name, type='signal')

        # Get the direct inputs to this signal
        try:
            binding = dfg.getBindings(term)
            if binding:
                for input_term in binding.getTerms():
                    input_signal = str(input_term)

                    # Skip temporary signals and constants
                    if input_signal.startswith('_') or input_signal.isdigit():
                        continue

                    # Add the input signal node
                    if input_signal not in self.data_flow_graph:
                        self.data_flow_graph.add_node(input_signal, type='signal')

                    # Add the edge (input affects output)
                    self.data_flow_graph.add_edge(input_signal, signal_name)
        except Exception as e:
            if self.verbose:
                print(
                    f"    Warning: Could not process binding for {signal_name}: {str(e)}"
                )

    def _build_cross_module_dataflow(self):
        """Build data flow connections across modules"""
        print("\nBuilding cross-module data flow connections...")

        # Debug: Print module info structure
        print(f"  DEBUG: Number of modules: {len(self.module_info)}")
        for module_name, module_data in self.module_info.items():
            print(
                f"  DEBUG: Module '{module_name}' has {len(module_data['instances'])} instances"
            )

            # Debug: Print all instances
            for instance_name, instance_data in module_data['instances'].items():
                print(
                    f"    DEBUG: Instance '{instance_name}' of type '{instance_data['module']}'"
                )
                print(
                    f"      DEBUG: Has {len(instance_data['connections'])} connections"
                )

        edges_added = 0

        # For each module instance, connect the ports
        for module_name, module_data in self.module_info.items():
            for instance_name, instance_data in module_data['instances'].items():
                inst_module = instance_data['module']

                # Check if the instantiated module is in our module info
                if inst_module not in self.module_info:
                    print(
                        f"  Warning: Module {inst_module} instantiated but not found in parsed files"
                    )
                    continue

                # Debug: Show port information
                print(
                    f"  DEBUG: Looking at connections for {instance_name} in {module_name}"
                )
                print(f"  DEBUG: This instance is of type {inst_module}")
                print(
                    f"  DEBUG: Available ports in {inst_module}: {list(self.module_info[inst_module]['ports'].keys())}"
                )

                # Connect ports between modules
                for port_name, connection in instance_data['connections'].items():
                    print(f"    DEBUG: Port connection: {port_name} -> {connection}")

                    if connection:
                        # The external signal connecting to this port
                        if port_name in self.module_info[inst_module]['ports']:
                            port_dir = self.module_info[inst_module]['ports'][
                                port_name
                            ]['direction']

                            print(
                                f"    DEBUG: Port {port_name} is in direction {port_dir}"
                            )

                            if port_dir == 'input':
                                # Data flows from parent module to instantiated module
                                source = f"{module_name}.{connection}"
                                target = f"{module_name}.{instance_name}.{port_name}"
                            else:
                                # Data flows from instantiated module to parent module
                                source = f"{module_name}.{instance_name}.{port_name}"
                                target = f"{module_name}.{connection}"

                            print(f"    DEBUG: Adding edge: {source} -> {target}")

                            # Add to data flow graph
                            self.data_flow_graph.add_node(source, type='signal')
                            self.data_flow_graph.add_node(target, type='signal')
                            self.data_flow_graph.add_edge(source, target)
                            edges_added += 1
                        else:
                            print(
                                f"    DEBUG: Port {port_name} not found in module {inst_module}"
                            )

        print(f"  Added {edges_added} cross-module connections to data flow graph")

    def _find_module_instances_direct(self):
        """Find module instances directly from file content"""
        print("\nDirectly extracting module instances from file content...")

        instance_count = 0

        # Find all Verilog files
        for root, _, files in os.walk(self.design_dir):
            for file in files:
                if file.endswith(('.v', '.sv')):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                        # Find module definitions
                        module_matches = re.findall(r'module\s+(\w+)', content)

                        # If this file contains module definitions
                        if module_matches:
                            parent_module = module_matches[
                                0
                            ]  # Assume first module found is the parent

                            if parent_module in self.module_info:
                                # Find instance declarations (module_name instance_name(...);)
                                # Simplified pattern - may need refinement
                                instance_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\);'
                                instances = re.findall(
                                    instance_pattern, content, re.DOTALL
                                )

                                for module_type, instance_name, port_list in instances:
                                    # Skip if this is not a module instantiation
                                    if (
                                        module_type not in self.module_info
                                        or module_type == parent_module
                                    ):
                                        continue

                                    print(
                                        f"  Found instance {instance_name} of module {module_type} in {parent_module}"
                                    )

                                    # Add to module info
                                    if (
                                        'instances'
                                        not in self.module_info[parent_module]
                                    ):
                                        self.module_info[parent_module][
                                            'instances'
                                        ] = {}

                                    self.module_info[parent_module]['instances'][
                                        instance_name
                                    ] = {'module': module_type, 'connections': {}}

                                    # Extract port connections
                                    connections = port_list.split(',')
                                    for conn in connections:
                                        conn = conn.strip()
                                        if (
                                            '.' in conn
                                        ):  # Named port connection (e.g., .port_name(signal_name))
                                            match = re.match(r'\.(\w+)\((\w+)\)', conn)
                                            if match:
                                                port_name = match.group(1)
                                                signal_name = match.group(2)
                                                self.module_info[parent_module][
                                                    'instances'
                                                ][instance_name]['connections'][
                                                    port_name
                                                ] = signal_name
                                        else:  # Positional port connection
                                            # For positional connections, we'd need module port order
                                            # This is simplified and may need refinement
                                            port_match = re.match(r'(\w+)', conn)
                                            if port_match:
                                                signal_name = port_match.group(1)
                                                # Use a placeholder port name for now
                                                placeholder_port = f"port_{len(self.module_info[parent_module]['instances'][instance_name]['connections'])}"
                                                self.module_info[parent_module][
                                                    'instances'
                                                ][instance_name]['connections'][
                                                    placeholder_port
                                                ] = signal_name

                                    instance_count += 1

                    except Exception as e:
                        print(f"  Error analyzing {file_path}: {str(e)}")

        print(f"  Found {instance_count} module instances through direct analysis")

    def _extract_port_info_direct(self):
        """Extract port information directly from file content"""
        print("\nDirectly extracting port information from file content...")

        port_count = 0

        # Find all Verilog files
        for root, _, files in os.walk(self.design_dir):
            for file in files:
                if file.endswith(('.v', '.sv')):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                        # Find module definitions with port lists
                        module_pattern = r'module\s+(\w+)\s*\((.*?)\);'
                        module_matches = re.findall(module_pattern, content, re.DOTALL)

                        for module_name, port_list in module_matches:
                            if module_name in self.module_info:
                                # Split and clean port list
                                ports = [p.strip() for p in port_list.split(',')]

                                # Find port direction declarations
                                input_pattern = (
                                    r'input\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)'
                                )
                                output_pattern = (
                                    r'output\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)'
                                )
                                inout_pattern = (
                                    r'inout\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)'
                                )

                                # Extract inputs
                                for msb, lsb, port_name in re.findall(
                                    input_pattern, content
                                ):
                                    if port_name in ports:
                                        width = f"{msb}:{lsb}" if msb and lsb else None
                                        self.module_info[module_name]['ports'][
                                            port_name
                                        ] = {'direction': 'input', 'width': width}
                                        # Add to primary signals
                                        self.primary_signals.add(
                                            f"{module_name}.{port_name}"
                                        )
                                        port_count += 1

                                # Extract outputs
                                for msb, lsb, port_name in re.findall(
                                    output_pattern, content
                                ):
                                    if port_name in ports:
                                        width = f"{msb}:{lsb}" if msb and lsb else None
                                        self.module_info[module_name]['ports'][
                                            port_name
                                        ] = {'direction': 'output', 'width': width}
                                        # Add to primary signals
                                        self.primary_signals.add(
                                            f"{module_name}.{port_name}"
                                        )
                                        port_count += 1

                                # Extract inouts
                                for msb, lsb, port_name in re.findall(
                                    inout_pattern, content
                                ):
                                    if port_name in ports:
                                        width = f"{msb}:{lsb}" if msb and lsb else None
                                        self.module_info[module_name]['ports'][
                                            port_name
                                        ] = {'direction': 'inout', 'width': width}
                                        # Add to primary signals
                                        self.primary_signals.add(
                                            f"{module_name}.{port_name}"
                                        )
                                        port_count += 1

                                print(
                                    f"  Extracted {len(self.module_info[module_name]['ports'])} ports for module {module_name}"
                                )

                    except Exception as e:
                        print(f"  Error extracting ports from {file_path}: {str(e)}")

        print(f"  Extracted {port_count} ports in total")

    def _identify_fsms(self):
        """Identify potential FSMs in the design"""
        print("\nIdentifying potential FSMs...")

        fsm_count = 0

        # Print the number of always blocks across all modules
        total_always_blocks = sum(
            len(m.get('always_blocks', [])) for m in self.module_info.values()
        )
        print(f"  Debug: Found {total_always_blocks} always blocks across all modules")

        # For each module, look for potential state registers
        for module_name, module_data in self.module_info.items():
            print(f"  Debug: Analyzing module {module_name} for FSMs")

            # Print the number of always blocks in this module
            always_blocks = module_data.get('always_blocks', [])
            print(f"    Debug: Module has {len(always_blocks)} always blocks")

            # Check for case statements directly from file content (since we may miss them in AST parsing)
            if module_name in [
                m.split('/')[-1].replace('.v', '') for m in self.file_info.keys()
            ]:
                for file_name, file_data in self.file_info.items():
                    if module_name in file_data.get('modules', []):
                        try:
                            with open(file_data['path'], 'r') as f:
                                content = f.read()

                                # Look for case statements - likely state machine transitions
                                case_statements = re.findall(
                                    r'case\s*\(\s*(\w+)\s*\)', content
                                )
                                if case_statements:
                                    print(
                                        f"    Debug: Found potential state variables in case statements: {', '.join(case_statements)}"
                                    )

                                    # Check if any variable looks like a state variable
                                    state_vars = [
                                        var
                                        for var in case_statements
                                        if 'state' in var.lower()
                                        or 'st' == var.lower()
                                        or 'st_' in var.lower()
                                        or 'fsm' in var.lower()
                                    ]

                                    if state_vars:
                                        print(
                                            f"    Debug: Likely state variables: {', '.join(state_vars)}"
                                        )

                                        # Add this as a potential FSM
                                        for state_var in state_vars:
                                            fsm_id = f"{module_name}_fsm_{state_var}"
                                            self.fsm_info[fsm_id] = {
                                                'module': module_name,
                                                'state_variable': state_var,
                                                'detection_method': 'case_statement',
                                                'clock_sensitive': True,  # Assumed - might need refinement
                                                'reset_sensitive': True,  # Assumed - might need refinement
                                            }
                                            fsm_count += 1

                                # Look for state register declarations
                                reg_declarations = re.findall(
                                    r'reg\s+(?:\[\s*\d+\s*:\s*\d+\s*\])?\s*(\w+)\s*;',
                                    content,
                                )
                                state_regs = [
                                    reg
                                    for reg in reg_declarations
                                    if 'state' in reg.lower()
                                    or 'st_' in reg.lower()
                                    or 'fsm' in reg.lower()
                                ]

                                if state_regs:
                                    print(
                                        f"    Debug: Found potential state registers: {', '.join(state_regs)}"
                                    )

                                    # Add each as a potential FSM
                                    for state_reg in state_regs:
                                        fsm_id = f"{module_name}_fsm_{state_reg}"
                                        if (
                                            fsm_id not in self.fsm_info
                                        ):  # Avoid duplicates
                                            self.fsm_info[fsm_id] = {
                                                'module': module_name,
                                                'state_variable': state_reg,
                                                'detection_method': 'reg_declaration',
                                                'clock_sensitive': True,  # Assumed - might need refinement
                                                'reset_sensitive': True,  # Assumed - might need refinement
                                            }
                                            fsm_count += 1

                                # Look for parameter definitions that could be state encodings
                                param_statements = re.findall(
                                    r'parameter\s+(\w+)\s*=\s*\d+', content
                                )
                                state_params = [
                                    param
                                    for param in param_statements
                                    if 'state' in param.lower()
                                    or 'st_' in param.lower()
                                    or 'idle' in param.lower()
                                    or 'busy' in param.lower()
                                    or 'wait' in param.lower()
                                    or 'done' in param.lower()
                                ]

                                if state_params:
                                    print(
                                        f"    Debug: Found potential state parameters: {', '.join(state_params)}"
                                    )

                                    if (
                                        len(state_params) >= 2
                                    ):  # Need at least 2 states to form an FSM
                                        fsm_id = f"{module_name}_fsm_params"
                                        self.fsm_info[fsm_id] = {
                                            'module': module_name,
                                            'state_encodings': state_params,
                                            'detection_method': 'parameter_definitions',
                                            'clock_sensitive': True,  # Assumed - might need refinement
                                            'reset_sensitive': True,  # Assumed - might need refinement
                                        }
                                        fsm_count += 1
                        except Exception as e:
                            print(f"    Error analyzing file for FSMs: {str(e)}")

            # Now check always blocks from the AST (our original approach)
            for idx, always_block in enumerate(always_blocks):
                print(f"    Debug: Examining always block {idx}")

                # Print sensitivity list details
                senslist = always_block.get('senslist', [])
                if senslist:
                    signals = [
                        f"{s.get('signal', 'unknown')}({s.get('type', 'unknown')})"
                        for s in senslist
                    ]
                    print(f"      Debug: Sensitivity list: {', '.join(signals)}")

                # Look for always blocks sensitive to clock and reset
                has_clock = False
                has_reset = False
                clock_signal = None
                reset_signal = None

                for sens in senslist:
                    signal = sens.get('signal', '')
                    signal_type = sens.get('type', '')

                    if signal_type in ('posedge', 'negedge'):
                        if 'clk' in signal.lower() or 'clock' in signal.lower():
                            has_clock = True
                            clock_signal = signal
                        if 'rst' in signal.lower() or 'reset' in signal.lower():
                            has_reset = True
                            reset_signal = signal

                if has_clock:
                    print(
                        f"      Debug: Identified clock-sensitive block with clock {clock_signal}"
                    )
                    # This is potentially an FSM sequential block
                    fsm_id = f"{module_name}_fsm_{idx}"
                    self.fsm_info[fsm_id] = {
                        'module': module_name,
                        'clock_signal': clock_signal,
                        'reset_signal': reset_signal,
                        'clock_sensitive': has_clock,
                        'reset_sensitive': has_reset,
                        'detection_method': 'always_block',
                    }
                    fsm_count += 1
                else:
                    print(f"      Debug: Not identified as clock-sensitive FSM block")

        # Print detailed FSM info
        if self.fsm_info:
            print("\n  Identified potential FSMs:")
            for fsm_id, fsm_data in self.fsm_info.items():
                detection = fsm_data.get('detection_method', 'unknown')
                print(f"    - {fsm_id} (detected via {detection})")
                for key, value in fsm_data.items():
                    if key != 'detection_method':
                        print(f"      {key}: {value}")

        print(f"  Identified {fsm_count} potential FSM(s) in the design")

        # If no FSMs found, do one more attempt with direct file content analysis
        if fsm_count == 0:
            print(
                "  No FSMs identified through structured analysis, attempting direct pattern matching..."
            )
            self._direct_fsm_pattern_match()

    def _direct_fsm_pattern_match(self):
        """Last-resort attempt to find FSM patterns in the raw file content"""
        fsm_count = 0

        # Common FSM patterns to look for in Verilog code
        patterns = [
            (
                r'reg\s+\[\s*\d+\s*:\s*\d+\s*\]\s*(\w+).*?(?:state|current|next|present|fsm|st)',
                "Multi-bit state register",
            ),
            (r'reg\s+(\w*state\w*)', "State register"),
            (r'parameter\s+(\w+STATE\w*)\s*=', "State parameter"),
            (r'localparam\s+(\w+STATE\w*)\s*=', "State localparam"),
            (r'case\s*\(\s*(\w*state\w*)\s*\)', "State in case statement"),
            (r'case\s*\(\s*(\w*current\w*)\s*\)', "Current state in case"),
            (r'case\s*\(\s*(\w*next\w*)\s*\)', "Next state in case"),
            (r'case\s*\(\s*(\w*present\w*)\s*\)', "Present state in case"),
            (
                r'case\s*\(\s*(\w+)\s*\).*?IDLE.*?BUSY',
                "State machine with IDLE/BUSY states",
            ),
            (
                r'always\s*@\s*\(\s*posedge.*?\).*?case',
                "Sequential block with case statement",
            ),
        ]

        # Look in each file
        for file_name, file_data in self.file_info.items():
            try:
                with open(file_data['path'], 'r') as f:
                    content = f.read()

                    print(f"  Analyzing {file_name} for FSM patterns")
                    for pattern, description in patterns:
                        matches = re.findall(
                            pattern, content, re.DOTALL | re.IGNORECASE
                        )
                        if matches:
                            # print(
                            #     f"    Found {len(matches)} matches for '{description}': {', '.join(matches[:3])}"
                            # )
                            print(
                                f"    Found {len(matches)} matches for '{description}'"
                            )

                            # Add as potential FSM
                            for match in matches:
                                if isinstance(match, str) and len(match) > 0:
                                    for module_name in file_data.get('modules', []):
                                        fsm_id = f"{module_name}_direct_fsm_{match}"
                                        self.fsm_info[fsm_id] = {
                                            'module': module_name,
                                            'pattern_match': description,
                                            'matched_text': match,
                                            'detection_method': 'direct_pattern',
                                        }
                                        fsm_count += 1
            except Exception as e:
                print(
                    f"  Error in direct FSM pattern matching for {file_name}: {str(e)}"
                )

        print(
            f"  Direct pattern matching identified {fsm_count} additional potential FSM(s)"
        )

    def _analyze_primary_signal_relationships(self):
        """Analyze relationships between primary (I/O) signals"""
        print("\nAnalyzing primary signal relationships...")

        # Debug: Show primary signals and graph nodes
        print(f"  Number of primary signals: {len(self.primary_signals)}")
        print(
            f"  Number of nodes in data flow graph: {self.data_flow_graph.number_of_nodes()}"
        )

        # Verify which primary signals are actually in the graph
        valid_primary_signals = set()
        missing_signals = set()
        for signal in self.primary_signals:
            if signal in self.data_flow_graph:
                valid_primary_signals.add(signal)
            else:
                missing_signals.add(signal)

        print(f"  Valid primary signals in graph: {len(valid_primary_signals)}")
        if missing_signals:
            print(
                f"  Warning: {len(missing_signals)} primary signals are not in the data flow graph"
            )
            if self.verbose:
                print(
                    f"  Missing signals: {', '.join(sorted(list(missing_signals)[:10]))}{'...' if len(missing_signals) > 10 else ''}"
                )

        # Only analyze paths between signals that exist in the graph
        primary_paths = []
        paths_analyzed = 0
        paths_found = 0

        for source in valid_primary_signals:
            for target in valid_primary_signals:
                if source != target:
                    paths_analyzed += 1
                    try:
                        if nx.has_path(self.data_flow_graph, source, target):
                            try:
                                paths = list(
                                    nx.all_shortest_paths(
                                        self.data_flow_graph, source, target
                                    )
                                )
                                primary_paths.append(
                                    {
                                        'source': source,
                                        'target': target,
                                        'path_length': len(paths[0]),
                                        'paths': paths,
                                    }
                                )
                                paths_found += 1
                            except nx.NetworkXNoPath:
                                # This shouldn't happen if has_path is true, but handle it just in case
                                pass
                    except Exception as e:
                        print(
                            f"  Error checking path from {source} to {target}: {str(e)}"
                        )

        print(f"  Analyzed {paths_analyzed} potential paths between primary signals")
        print(f"  Found {paths_found} direct relationship(s) between primary signals")

        # Print some example paths
        if primary_paths:
            print("\n  Example paths between primary signals:")
            for i, path_info in enumerate(primary_paths[:3]):  # Show first 3 examples
                print(
                    f"    Path {i+1}: {path_info['source']} -> {path_info['target']} (length: {path_info['path_length']})"
                )
                for j, path in enumerate(
                    path_info['paths'][:2]
                ):  # Show up to 2 paths for each relationship
                    print(f"      Path variant {j+1}: {' -> '.join(path)}")

    def _print_summary(self):
        """Print a summary of the analyzed RTL design and store it for later use"""
        print("\n" + "=" * 80)
        print("RTL ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"\nProcessed {len(self.file_info)} file(s)")
        print(f"Found {len(self.module_info)} module(s)")
        print(f"Identified {len(self.primary_signals)} primary signal(s)")
        print(
            f"Built data flow graph with {self.data_flow_graph.number_of_nodes()} nodes and {self.data_flow_graph.number_of_edges()} edges"
        )
        print(f"Detected {len(self.fsm_info)} potential FSM(s)")

        # Print modules and their ports
        print("\nModules and primary I/O signals:")
        for module_name, module_data in self.module_info.items():
            print(f"  Module: {module_name}")
            for port_name, port_data in module_data['ports'].items():
                width_str = f"[{port_data['width']}]" if port_data['width'] else ""
                print(f"    {port_data['direction']} {width_str} {port_name}")

                # Consider this a primary signal
                qualified_name = f"{module_name}.{port_name}"
                if qualified_name not in self.primary_signals:
                    self.primary_signals.add(qualified_name)

        # Print module hierarchy
        print("\nModule hierarchy:")
        for module_name, module_data in self.module_info.items():
            print(f"  Module: {module_name}")
            for instance_name, instance_data in module_data['instances'].items():
                print(f"    Instantiates: {instance_data['module']} as {instance_name}")

        # Store summary information in object attributes
        self.summary = {
            'file_count': len(self.file_info),
            'module_count': len(self.module_info),
            'primary_signal_count': len(self.primary_signals),
            'graph_nodes': self.data_flow_graph.number_of_nodes(),
            'graph_edges': self.data_flow_graph.number_of_edges(),
            'fsm_count': len(self.fsm_info),
        }

    def _enhanced_signal_analysis(self):
        """Perform enhanced analysis of signals for SVA generation"""
        print("\n" + "=" * 80)
        print("ENHANCED SIGNAL ANALYSIS")
        print("=" * 80)

        # If we have primary signals, analyze them
        if self.primary_signals:
            print(f"\nPrimary signals identified: {len(self.primary_signals)}")
            for signal in sorted(list(self.primary_signals)):
                print(f"  - {signal}")

            # Perform generic protocol pattern analysis
            self._analyze_protocol_patterns()
        else:
            print("\nNo primary signals identified - performing direct RTL analysis")

            # Perform direct analysis on the RTL files
            self._direct_rtl_analysis()

    def _analyze_protocol_patterns(self):
        """Generic analysis of protocol patterns in the design"""
        print("\nAnalyzing protocol patterns in the design...")

        # Initialize protocol patterns dictionary
        self.protocol_patterns = {}

        # Look through all modules to identify common patterns
        for module_name, module_data in self.module_info.items():
            print(f"  Analyzing module {module_name} for protocol patterns")

            # Initialize module pattern data
            module_patterns = {
                'clock_signals': [],
                'reset_signals': [],
                'data_signals': [],
                'control_signals': [],
                'handshaking_signals': [],
            }

            # Scan ports to identify common signal types based on naming conventions
            for port_name, port_data in module_data.get('ports', {}).items():
                port_name_lower = port_name.lower()

                # Clock signals
                if 'clk' in port_name_lower or 'clock' in port_name_lower:
                    module_patterns['clock_signals'].append(port_name)

                # Reset signals
                elif 'rst' in port_name_lower or 'reset' in port_name_lower:
                    module_patterns['reset_signals'].append(port_name)

                # Data signals
                elif 'data' in port_name_lower or '_d' == port_name_lower[-2:]:
                    module_patterns['data_signals'].append(port_name)

                # Control signals
                elif any(
                    ctrl in port_name_lower for ctrl in ['en', 'sel', 'ctrl', 'control']
                ):
                    module_patterns['control_signals'].append(port_name)

                # Handshaking signals
                elif any(
                    hs in port_name_lower for hs in ['valid', 'ready', 'ack', 'req']
                ):
                    module_patterns['handshaking_signals'].append(port_name)

            # Store protocol patterns for this module if any were found
            if any(len(signals) > 0 for signals in module_patterns.values()):
                self.protocol_patterns[module_name] = module_patterns

                if self.verbose:
                    print(f"  Found protocol patterns in module {module_name}:")
                    for pattern_type, signals in module_patterns.items():
                        if signals:
                            print(f"    {pattern_type}: {', '.join(signals)}")

                # Generate generic verification suggestions based on patterns
                self._generate_verification_suggestions(module_name, module_patterns)

    def _generate_verification_suggestions(self, module_name, patterns):
        """Generate generic verification suggestions based on identified patterns"""
        suggestions = []

        # Reset behavior verification if reset signals exist
        if patterns['reset_signals']:
            suggestion = {
                'type': 'reset_behavior',
                'module': module_name,
                'description': 'Verify proper reset behavior',
                'signals': patterns['reset_signals'],
                'suggestion': 'Check all outputs go to defined state when reset is active',
            }
            suggestions.append(suggestion)

        # Clock domain crossing if multiple clock signals exist
        if len(patterns['clock_signals']) > 1:
            suggestion = {
                'type': 'clock_domain_crossing',
                'module': module_name,
                'description': 'Check clock domain crossing',
                'signals': patterns['clock_signals'],
                'suggestion': 'Verify signals crossing clock domains use proper synchronization',
            }
            suggestions.append(suggestion)

        # Data stability verification if data and control signals exist
        if patterns['data_signals'] and (
            patterns['control_signals'] or patterns['clock_signals']
        ):
            control_signals = patterns['control_signals'] or patterns['clock_signals']
            suggestion = {
                'type': 'data_stability',
                'module': module_name,
                'description': 'Verify data stability',
                'signals': patterns['data_signals']
                + control_signals[:1],  # Use the first control signal
                'suggestion': 'Check data signals maintain stable values when being sampled',
            }
            suggestions.append(suggestion)

        # Handshaking protocol verification
        if patterns['handshaking_signals']:
            req_signals = [
                s
                for s in patterns['handshaking_signals']
                if 'req' in s.lower() or 'valid' in s.lower()
            ]
            ack_signals = [
                s
                for s in patterns['handshaking_signals']
                if 'ack' in s.lower() or 'ready' in s.lower()
            ]

            if req_signals and ack_signals:
                suggestion = {
                    'type': 'handshaking_protocol',
                    'module': module_name,
                    'description': 'Verify request-acknowledge handshaking',
                    'signals': req_signals + ack_signals,
                    'suggestion': 'Check proper sequencing of request and acknowledge signals',
                }
                suggestions.append(suggestion)

        # Store verification suggestions for this module
        if suggestions:
            print(
                f"  Generated {len(suggestions)} verification suggestions for module {module_name}:\n{suggestions}"
            )
            self.verification_suggestions.append(
                {'module': module_name, 'suggestions': suggestions}
            )

    def _direct_rtl_analysis(self):
        """Perform direct analysis of RTL files when module parsing fails"""
        print("\nPerforming direct analysis of RTL files...")

        # Find all Verilog files
        verilog_files = []
        for root, _, files in os.walk(self.design_dir):
            for file in files:
                if file.endswith(('.v', '.sv')):
                    verilog_files.append(os.path.join(root, file))

        # Extract information directly from file content
        for file_path in verilog_files:
            filename = os.path.basename(file_path)
            print(f"\nAnalyzing file: {filename}")

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Look for module definitions
                module_matches = re.findall(r'module\s+(\w+)', content)
                if module_matches:
                    print(f"  Found module(s): {', '.join(module_matches)}")

                    # Find ports by looking for input/output declarations
                    input_ports = re.findall(
                        r'input\s+(?:\[\s*\d+\s*:\s*\d+\s*\])?\s*(\w+)', content
                    )
                    output_ports = re.findall(
                        r'output\s+(?:\[\s*\d+\s*:\s*\d+\s*\])?\s*(\w+)', content
                    )

                    if input_ports:
                        print(f"  Input ports: {', '.join(input_ports)}")
                        # Add to primary signals
                        for port in input_ports:
                            for module in module_matches:
                                self.primary_signals.add(f"{module}.{port}")

                    if output_ports:
                        print(f"  Output ports: {', '.join(output_ports)}")
                        # Add to primary signals
                        for port in output_ports:
                            for module in module_matches:
                                self.primary_signals.add(f"{module}.{port}")

                    # Find always blocks - potential state machine indicators
                    always_blocks = re.findall(r'always\s*@\s*\((.*?)\)', content)
                    if always_blocks:
                        print(f"  Found {len(always_blocks)} always block(s)")

                        # Check for state machines
                        potential_fsms = []
                        for i, senslist in enumerate(always_blocks):
                            is_clocked = 'posedge' in senslist or 'negedge' in senslist
                            has_reset = 'reset' in senslist or 'rst' in senslist

                            if is_clocked:
                                # Find the clock signal
                                clock_match = re.search(
                                    r'(pos|neg)edge\s+(\w+)', senslist
                                )
                                clock_signal = (
                                    clock_match.group(2) if clock_match else "unknown"
                                )

                                # This could be an FSM
                                potential_fsms.append(
                                    {
                                        'id': f"fsm_{i}",
                                        'clock': clock_signal,
                                        'has_reset': has_reset,
                                    }
                                )

                        if potential_fsms:
                            print(
                                f"  Identified {len(potential_fsms)} potential FSM(s)"
                            )
                            for fsm in potential_fsms:
                                print(
                                    f"    - FSM {fsm['id']} with clock {fsm['clock']}, reset: {'Yes' if fsm['has_reset'] else 'No'}"
                                )

                    # Check for case statements - likely state machines
                    case_statements = re.findall(
                        r'case\s*\((.*?)\)(.*?)endcase', content, re.DOTALL
                    )
                    if case_statements:
                        print(
                            f"  Found {len(case_statements)} case statement(s) - potential state machine transitions"
                        )

                        for i, (case_var, case_body) in enumerate(case_statements):
                            states = re.findall(r'\s*(\w+)\s*:', case_body)
                            if states:
                                print(
                                    f"    - Case statement {i+1} for variable {case_var} has {len(states)} state(s)"
                                )
                                if len(states) <= 10:  # Don't print too many states
                                    print(f"      States: {', '.join(states)}")

                # Look for signal assignments that could indicate important relationships
                assignments = re.findall(r'assign\s+(\w+)\s*=\s*(.*?);', content)
                if assignments:
                    print(f"  Found {len(assignments)} signal assignment(s)")

                    # Look for interesting assignments
                    for i, (lhs, rhs) in enumerate(assignments[:5]):  # Show first 5
                        print(f"    - {lhs} = {rhs}")

                        # Check if this involves a primary signal
                        for sig in self.primary_signals:
                            if sig.endswith(f".{lhs}") or sig.split(".")[-1] in rhs:
                                print(
                                    f"      Note: This assignment involves primary signal {sig}"
                                )

            except Exception as e:
                print(f"  Error analyzing {filename}: {str(e)}")

        # After direct analysis, generate verification suggestions
        if self.primary_signals:
            print("\nVerification suggestions based on direct analysis:")

            # Group signals by type
            clock_signals = [
                s
                for s in self.primary_signals
                if 'clk' in s.lower() or 'clock' in s.lower()
            ]
            reset_signals = [
                s
                for s in self.primary_signals
                if 'rst' in s.lower() or 'reset' in s.lower()
            ]
            data_signals = [s for s in self.primary_signals if 'data' in s.lower()]
            control_signals = [
                s
                for s in self.primary_signals
                if 'en' in s.lower() or 'valid' in s.lower() or 'ready' in s.lower()
            ]

            # Generate SVA suggestions
            if clock_signals and reset_signals:
                print("  1. Reset behavior verification:")
                print(
                    f"     - Check that all relevant outputs are properly reset when {reset_signals[0]} is asserted"
                )
                print(f"     - Use clock {clock_signals[0]} for synchronization")

            if data_signals and clock_signals:
                print("  2. Data integrity verification:")
                print(
                    f"     - Verify that data signals {', '.join(data_signals[:3])} maintain valid values during operation"
                )

            if control_signals:
                print("  3. Control signal protocol verification:")
                print(
                    f"     - Check proper sequencing of control signals {', '.join(control_signals[:3])}"
                )

            # Specific to UART
            uart_signals = [
                s
                for s in self.primary_signals
                if 'uart' in s.lower()
                or 'ser_' in s.lower()
                or 'rx' in s.lower()
                or 'tx' in s.lower()
            ]
            if uart_signals:
                print("  4. UART protocol verification:")
                print("     - Verify start/stop bit timing")
                print("     - Check baud rate generation accuracy")
                print("     - Verify data sampling at the correct bit time")

    def get_analysis_results(self):
        """Get all analysis results in a structured format"""
        return {
            'files': self.file_info,
            'modules': self.module_info,
            'fsms': self.fsm_info,
            'primary_signals': list(self.primary_signals),
            'signal_types': self.signal_type_info,
            'protocol_patterns': getattr(self, 'protocol_patterns', {}),
            'verification_suggestions': getattr(self, 'verification_suggestions', []),
            'summary': getattr(self, 'summary', {}),
            # Add new information
            'control_flow': self.control_flow,
            'signal_attributes': self.signal_type_info,
        }

    def _extract_control_flow(self, module_name, file_path):
        """Extract control flow structures from the module"""
        print(f"\nExtracting control flow for module {module_name}...")

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Initialize control flow data structure if not exists
            if module_name not in self.control_flow:
                self.control_flow[module_name] = {
                    'if_statements': [],
                    'case_statements': [],
                    'loops': [],
                }

            # Extract if statements with their conditions - simplified pattern
            if_pattern = r'if\s*\((.*?)\)'
            if_matches = re.findall(if_pattern, content)

            for i, condition in enumerate(if_matches):
                self.control_flow[module_name]['if_statements'].append(
                    {'id': f"{module_name}_if_{i}", 'condition': condition.strip()}
                )

            print(
                f"  Found {len(self.control_flow[module_name]['if_statements'])} if statements"
            )

            # Extract case statements - simpler approach
            case_pattern = r'case\s*\(\s*(.+?)\s*\)'
            case_matches = re.findall(case_pattern, content)

            print(
                f"  Debug: Found {len(case_matches)} case statement matches in {module_name}"
            )

            for i, case_expr in enumerate(case_matches):
                self.control_flow[module_name]['case_statements'].append(
                    {
                        'id': f"{module_name}_case_{i}",
                        'case_expression': case_expr.strip(),
                    }
                )

            print(
                f"  Found {len(self.control_flow[module_name]['case_statements'])} case statements"
            )

            # Extract loop structures (for, while) - simpler pattern
            for_pattern = r'for\s*\('
            for_matches = re.findall(for_pattern, content)

            for i in range(len(for_matches)):
                self.control_flow[module_name]['loops'].append(
                    {'id': f"{module_name}_for_{i}", 'type': 'for'}
                )

            while_pattern = r'while\s*\('
            while_matches = re.findall(while_pattern, content)

            for i in range(len(while_matches)):
                self.control_flow[module_name]['loops'].append(
                    {'id': f"{module_name}_while_{i}", 'type': 'while'}
                )

            print(
                f"  Found {len(self.control_flow[module_name]['loops'])} loop structures"
            )

        except Exception as e:
            print(f"  Error extracting control flow for {module_name}: {str(e)}")

    def _extract_assignments(self, module_name, file_path):
        """Extract assignment relationships between signals"""
        print(f"\nExtracting assignments for module {module_name}...")

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Initialize assignment data structure
            if 'assignments' not in self.module_info[module_name]:
                self.module_info[module_name]['assignments'] = []  # Clear previous data

            # Extract continuous assignments (assign statements)
            assign_pattern = (
                r'assign\s+(\w+(?:\s*\[\s*\d+(?::\d+)?\s*\])?)\s*=\s*(.*?);'
            )
            assign_matches = re.findall(assign_pattern, content)

            for lhs, rhs in assign_matches:
                # Extract signals used in the RHS
                rhs_signals = re.findall(r'\b(\w+)\b', rhs)
                # Filter out keywords and numbers
                rhs_signals = [
                    s
                    for s in rhs_signals
                    if not s.isdigit()
                    and s not in ['and', 'or', 'not', 'xor', 'if', 'else']
                ]

                self.module_info[module_name]['assignments'].append(
                    {
                        'type': 'continuous',
                        'lhs': lhs.strip(),
                        'rhs': rhs.strip(),
                        'rhs_signals': rhs_signals,
                    }
                )

            # Extract procedural assignments in always blocks
            always_blocks = re.findall(
                r'always\s*@\s*\((.*?)\)(.*?)(?=always|module|endmodule|$)',
                content,
                re.DOTALL,
            )

            for i, (sensitivity, always_body) in enumerate(always_blocks):
                # Check if it's a clocked block
                is_clocked = 'posedge' in sensitivity or 'negedge' in sensitivity

                # Extract assignments within the always block
                block_assigns = re.findall(
                    r'(\w+(?:\s*\[\s*\d+(?::\d+)?\s*\])?)\s*(<=|=)\s*(.*?);',
                    always_body,
                )

                for lhs, assign_type, rhs in block_assigns:
                    assignment_type = (
                        'non_blocking' if assign_type == '<=' else 'blocking'
                    )

                    # Extract signals used in the RHS
                    rhs_signals = re.findall(r'\b(\w+)\b', rhs)
                    # Filter out keywords and numbers
                    rhs_signals = [
                        s
                        for s in rhs_signals
                        if not s.isdigit()
                        and s not in ['and', 'or', 'not', 'xor', 'if', 'else']
                    ]

                    self.module_info[module_name]['assignments'].append(
                        {
                            'type': assignment_type,
                            'lhs': lhs.strip(),
                            'rhs': rhs.strip(),
                            'rhs_signals': rhs_signals,
                            'in_clocked_block': is_clocked,
                            'always_block': i,
                        }
                    )

            total_assignments = len(self.module_info[module_name]['assignments'])
            print(f"  Found {total_assignments} assignments")

            # Count by type
            continuous_count = sum(
                1
                for a in self.module_info[module_name]['assignments']
                if a['type'] == 'continuous'
            )
            blocking_count = sum(
                1
                for a in self.module_info[module_name]['assignments']
                if a['type'] == 'blocking'
            )
            non_blocking_count = sum(
                1
                for a in self.module_info[module_name]['assignments']
                if a['type'] == 'non_blocking'
            )

            print(
                f"  Assignment types: {continuous_count} continuous, {blocking_count} blocking, {non_blocking_count} non-blocking"
            )

            # Print some examples
            if self.module_info[module_name]['assignments'] and self.verbose:
                print("\n  Example assignments:")
                for i, assignment in enumerate(
                    self.module_info[module_name]['assignments'][:3]
                ):  # First 3
                    print(
                        f"    {i+1}. {assignment['lhs']} {' <= ' if assignment['type'] == 'non_blocking' else ' = '}{assignment['rhs']}"
                    )
                    print(f"       Type: {assignment['type']}")
                    if 'in_clocked_block' in assignment:
                        print(
                            f"       In clocked block: {'Yes' if assignment['in_clocked_block'] else 'No'}"
                        )
                    print(
                        f"       Signals used: {', '.join(assignment['rhs_signals'])}"
                    )

        except Exception as e:
            print(f"  Error extracting assignments for {module_name}: {str(e)}")

    def _extract_signal_attributes(self, module_name, file_path):
        """Extract detailed attributes of all signals in the module"""
        print(f"\nExtracting signal attributes for module {module_name}...")

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Initialize signal attributes data structure
            self.signal_type_info[module_name] = {}

            # Extract wire declarations
            wire_pattern = (
                r'wire\s+(?:signed\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)\s*;'
            )
            wire_matches = re.findall(wire_pattern, content)

            for msb, lsb, name in wire_matches:
                width = f"{msb}:{lsb}" if msb and lsb else "1"
                self.signal_type_info[module_name][name] = {
                    'type': 'wire',
                    'width': width,
                    'signed': 'signed' in content.split(name)[0].split('\n')[-1],
                    'is_port': name in self.module_info[module_name].get('ports', {}),
                }

            # Extract reg declarations
            reg_pattern = (
                r'reg\s+(?:signed\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)\s*;'
            )
            reg_matches = re.findall(reg_pattern, content)

            for msb, lsb, name in reg_matches:
                width = f"{msb}:{lsb}" if msb and lsb else "1"
                self.signal_type_info[module_name][name] = {
                    'type': 'reg',
                    'width': width,
                    'signed': 'signed' in content.split(name)[0].split('\n')[-1],
                    'is_port': name in self.module_info[module_name].get('ports', {}),
                }

                # Look for initial value assignments
                initial_pattern = r'initial\s+' + name + r'\s*=\s*(.*?);'
                initial_match = re.search(initial_pattern, content)
                if initial_match:
                    self.signal_type_info[module_name][name]['initial_value'] = (
                        initial_match.group(1).strip()
                    )

            # Extract parameters
            param_pattern = r'parameter\s+(\w+)\s*=\s*(.*?);'
            param_matches = re.findall(param_pattern, content)

            for name, value in param_matches:
                self.signal_type_info[module_name][name] = {
                    'type': 'parameter',
                    'value': value.strip(),
                    'is_port': False,
                }

            # Extract localparams
            localparam_pattern = r'localparam\s+(\w+)\s*=\s*(.*?);'
            localparam_matches = re.findall(localparam_pattern, content)

            for name, value in localparam_matches:
                self.signal_type_info[module_name][name] = {
                    'type': 'localparam',
                    'value': value.strip(),
                    'is_port': False,
                }

            # SystemVerilog types (logic, bit, etc.)
            sv_types = [
                'logic',
                'bit',
                'byte',
                'shortint',
                'int',
                'longint',
                'integer',
                'time',
            ]
            for sv_type in sv_types:
                sv_pattern = rf'{sv_type}\s+(?:signed\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)\s*;'
                sv_matches = re.findall(sv_pattern, content)

                for msb, lsb, name in sv_matches:
                    width = f"{msb}:{lsb}" if msb and lsb else "1"
                    self.signal_type_info[module_name][name] = {
                        'type': sv_type,
                        'width': width,
                        'signed': 'signed' in content.split(name)[0].split('\n')[-1],
                        'is_port': name
                        in self.module_info[module_name].get('ports', {}),
                    }

            total_signals = len(self.signal_type_info[module_name])
            print(f"  Found {total_signals} signals/parameters")

            # Count by type
            type_counts = {}
            for signal, info in self.signal_type_info[module_name].items():
                signal_type = info['type']
                type_counts[signal_type] = type_counts.get(signal_type, 0) + 1

            print("  Signal types:")
            for sig_type, count in type_counts.items():
                print(f"    {sig_type}: {count}")

            # Print some examples
            if self.signal_type_info[module_name] and self.verbose:
                print("\n  Example signals:")
                i = 0
                for name, info in self.signal_type_info[module_name].items():
                    if i >= 3:  # First 3 examples
                        break
                    print(f"    {name}:")
                    for attr, value in info.items():
                        print(f"      {attr}: {value}")
                    i += 1

        except Exception as e:
            print(f"  Error extracting signal attributes for {module_name}: {str(e)}")

    def _print_expanded_summary(self):
        """Print an expanded summary including the new analysis information"""
        print("\n" + "=" * 80)
        print("EXPANDED RTL ANALYSIS SUMMARY")
        print("=" * 80)

        # Debug: Print what's in the control flow dictionary
        print("\nDebug - Control Flow Data:")
        for module_name, data in self.control_flow.items():
            print(f"  Module {module_name}:")
            print(f"    if_statements: {len(data.get('if_statements', []))}")
            print(f"    case_statements: {len(data.get('case_statements', []))}")
            print(f"    loops: {len(data.get('loops', []))}")

        # Control flow statistics
        total_if_stmts = sum(
            len(data.get('if_statements', []))
            for module_name, data in self.control_flow.items()
        )
        total_case_stmts = sum(
            len(data.get('case_statements', []))
            for module_name, data in self.control_flow.items()
        )
        total_loops = sum(
            len(data.get('loops', []))
            for module_name, data in self.control_flow.items()
        )

        print(f"\nControl Flow Statistics:")
        print(f"  Total if statements: {total_if_stmts}")
        print(f"  Total case statements: {total_case_stmts}")
        print(f"  Total loop structures: {total_loops}")

        # Assignments statistics
        total_assignments = 0
        continuous_assigns = 0
        blocking_assigns = 0
        non_blocking_assigns = 0

        for module_data in self.module_info.values():
            for assign in module_data.get('assignments', []):
                total_assignments += 1
                if assign['type'] == 'continuous':
                    continuous_assigns += 1
                elif assign['type'] == 'blocking':
                    blocking_assigns += 1
                elif assign['type'] == 'non_blocking':
                    non_blocking_assigns += 1

        print(f"\nAssignment Statistics:")
        print(f"  Total assignments: {total_assignments}")
        print(f"  Continuous assignments: {continuous_assigns}")
        print(f"  Blocking assignments: {blocking_assigns}")
        print(f"  Non-blocking assignments: {non_blocking_assigns}")

        # Signal type statistics
        total_signals = 0
        signal_types = {}

        for module_signals in self.signal_type_info.values():
            total_signals += len(module_signals)
            for signal_info in module_signals.values():
                sig_type = signal_info.get('type', 'unknown')
                signal_types[sig_type] = signal_types.get(sig_type, 0) + 1

        print(f"\nSignal Statistics:")
        print(f"  Total signals: {total_signals}")
        print("  By type:")
        for sig_type, count in signal_types.items():
            print(f"    {sig_type}: {count}")

        # Module complexity metrics
        print(f"\nModule Complexity Metrics:")
        for module_name in sorted(self.module_info.keys()):
            num_ports = len(self.module_info[module_name].get('ports', {}))
            num_assigns = len(self.module_info[module_name].get('assignments', []))
            num_if_stmts = len(
                self.control_flow.get(module_name, {}).get('if_statements', [])
            )
            num_case_stmts = len(
                self.control_flow.get(module_name, {}).get('case_statements', [])
            )
            num_signals = len(self.signal_type_info.get(module_name, {}))

            complexity_score = (
                num_ports + num_assigns + (2 * num_if_stmts) + (2 * num_case_stmts)
            )

            print(f"  Module: {module_name}")
            print(f"    Ports: {num_ports}")
            print(f"    Signals: {num_signals}")
            print(f"    Assignments: {num_assigns}")
            print(f"    Control structures: {num_if_stmts + num_case_stmts}")
            print(f"    Complexity score: {complexity_score}")


def process_files_in_order(design_dir):
    """Process files in a specific order to handle dependencies"""
    # First, find all Verilog files
    verilog_files = []
    for root, _, files in os.walk(design_dir):
        for file in files:
            if file.endswith(('.v', '.sv')):
                verilog_files.append(os.path.join(root, file))

    # Read all files to find include relationships
    include_map = {}
    module_map = {}

    for file_path in verilog_files:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find includes
        includes = re.findall(r'`include\s+"([^"]+)"', content)
        include_map[file_path] = includes

        # Find module definitions
        modules = re.findall(r'module\s+(\w+)', content)
        for module in modules:
            module_map[module] = file_path

    # Build dependency graph
    dep_graph = nx.DiGraph()

    # Add all files as nodes
    for file_path in verilog_files:
        dep_graph.add_node(file_path)

    # Add dependencies based on includes
    for file_path, includes in include_map.items():
        for include in includes:
            include_path = os.path.join(os.path.dirname(file_path), include)
            if include_path in verilog_files:
                dep_graph.add_edge(file_path, include_path)  # file depends on include

    # Add dependencies based on module instantiations
    for file_path in verilog_files:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find module instantiations (simplified)
        inst_pattern = r'(\w+)\s+(\w+)\s*\('
        for module_type, instance_name in re.findall(inst_pattern, content):
            if module_type in module_map and module_map[module_type] != file_path:
                dep_graph.add_edge(
                    file_path, module_map[module_type]
                )  # file depends on module definition

    # Sort files in dependency order (process included files first)
    try:
        # Use topological sort to order files
        sorted_files = list(nx.topological_sort(dep_graph))
        # Reverse to get leaves (most basic files) first
        sorted_files.reverse()
        return sorted_files
    except nx.NetworkXUnfeasible:
        print("Warning: Cyclic dependencies detected in Verilog files")
        return verilog_files  # Fall back to unsorted list


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Verilog RTL code to extract design information'
    )
    parser.add_argument(
        '--design_dir',
        type=str,
        required=True,
        help='Directory containing Verilog files',
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument(
        '--process_single_file',
        action='store_true',
        help='Process files one by one instead of using preprocessing',
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PREPROCESSING RTL FILES")
    print("=" * 80)

    sorted_files = process_files_in_order(args.design_dir)
    print(f"Found {len(sorted_files)} Verilog files, sorted in dependency order:")
    for i, file_path in enumerate(sorted_files):
        print(f"  {i+1}. {os.path.basename(file_path)}")

    analyzer = RTLAnalyzer(args.design_dir, args.verbose)
    analyzer.analyze_design()


if __name__ == "__main__":
    main()
