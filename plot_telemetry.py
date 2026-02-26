"""
Script to create plots of telemetry points from HDF5 files.

This script uses TelemetryReader from read_telemetry_hdf5.py to read telemetry data
and create plots of specified telemetry points. If no points are specified,
it creates a default stack plot of temperature measurements from beacon packets.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import TelemetryReader from read_telemetry_hdf5
from read_telemetry_hdf5 import TelemetryReader


def get_default_temperature_fields():
    """
    Get the list of default temperature field keywords to search for.
    Each keyword should appear in the field name along with 'temp'.
    
    Returns:
        list: List of keywords to search for (field must contain both keyword and 'temp')
    """
    return [
        'batt',      # All battery temperatures (e.g., beac_batt1_temp, beac_batt2_temp, beac_batt_board_temp)
        'cdh',       # CDH temperature (e.g., beac_cdh_temp)
        'eps',       # EPS board temperature (e.g., beac_eps_temp)
        'solar',     # Both solar panel temperatures (e.g., beac_solar_panel_1_temp, beac_solar_panel_2_temp)
        'ifb',       # IFB temperature (e.g., beac_ifb_temp)
        'uhf',       # UHF radio temperature (e.g., beac_uhf_temp)
        'adcs',      # ADCS motor 1 temperature (e.g., beac_adcs_motor_1_temp)
    ]


def find_matching_fields(available_fields, keywords, case_sensitive=False):
    """
    Find fields that contain both a keyword and 'temp' (for temperature fields).
    Special handling for 'adcs' keyword to ensure it matches motor 1 specifically.
    
    Args:
        available_fields (list): List of available field names
        keywords (list): List of keywords to search for (field must contain keyword AND 'temp')
        case_sensitive (bool): Whether matching should be case-sensitive
        
    Returns:
        list: List of matching field names
    """
    matching_fields = []
    
    for keyword in keywords:
        keyword_lower = keyword if case_sensitive else keyword.lower()
        temp_lower = 'temp' if case_sensitive else 'temp'
        
        for field in available_fields:
            field_to_match = field if case_sensitive else field.lower()
            
            # Special handling for ADCS - must contain 'adcs', 'motor', '1', and 'temp'
            if keyword_lower == 'adcs':
                if ('adcs' in field_to_match and 'motor' in field_to_match and 
                    '1' in field_to_match and temp_lower in field_to_match):
                    if field not in matching_fields:
                        matching_fields.append(field)
            else:
                # For other keywords, field must contain both the keyword and 'temp'
                if keyword_lower in field_to_match and temp_lower in field_to_match:
                    if field not in matching_fields:
                        matching_fields.append(field)
    
    return matching_fields


def filter_outliers(data, lower_percentile=0.5, upper_percentile=99.5):
    """
    Filter out outliers from data using percentile-based method.
    
    Args:
        data (numpy.ndarray): Data array to filter
        lower_percentile (float): Lower percentile threshold (default: 0.5)
        upper_percentile (float): Upper percentile threshold (default: 99.5)
    
    Returns:
        numpy.ndarray: Boolean mask where True indicates valid (non-outlier) values
    """
    if len(data) == 0:
        return np.array([], dtype=bool)
    
    # Calculate percentiles
    lower_bound = np.nanpercentile(data, lower_percentile)
    upper_bound = np.nanpercentile(data, upper_percentile)
    
    # Create mask for values within bounds
    mask = (data >= lower_bound) & (data <= upper_bound)
    
    return mask


def create_stack_plot(reader, fields, time_field='beac_time_since_boot', 
                      packet_type='beacon', title='Telemetry Stack Plot'):
    """
    Create a stack plot of multiple telemetry fields.
    
    Args:
        reader (TelemetryReader): The telemetry reader instance
        fields (list): List of field names to plot
        time_field (str): Name of the time field to use for x-axis
        packet_type (str): Packet type to filter by
        title (str): Title for the plot
    """
    # Filter by packet type
    filtered_data = reader.filter_by_packet_type(packet_type, case_sensitive=False)
    
    if len(filtered_data) == 0:
        print(f"Warning: No {packet_type} packets found. Cannot create plot.")
        return
    
    # Get available fields
    if hasattr(filtered_data, 'dtype') and hasattr(filtered_data.dtype, 'names'):
        available_fields = filtered_data.dtype.names
    else:
        # Columnar format
        available_fields = list(filtered_data.data_dict.keys())
    
    # Find time field
    time_fields = [f for f in available_fields if time_field.lower() in f.lower()]
    if not time_fields:
        print(f"Warning: Time field '{time_field}' not found. Available fields: {available_fields[:10]}...")
        return
    
    time_data = filtered_data[time_fields[0]]
    
    # Filter out NaN values in time
    valid_time_mask = ~np.isnan(time_data)
    time_valid = time_data[valid_time_mask]
    
    if len(time_valid) == 0:
        print("Warning: No valid time data found.")
        return
    
    # Create figure with subplots
    num_fields = len(fields)
    if num_fields == 0:
        print("Warning: No fields to plot.")
        return
    
    fig, axes = plt.subplots(num_fields, 1, figsize=(12, 2 * num_fields), sharex=True)
    if num_fields == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot each field
    for i, field_name in enumerate(fields):
        if field_name not in available_fields:
            print(f"Warning: Field '{field_name}' not found in available fields. Skipping.")
            axes[i].text(0.5, 0.5, f"Field '{field_name}' not found", 
                        transform=axes[i].transAxes, ha='center', va='center')
            continue
        
        field_data = filtered_data[field_name]
        field_valid = field_data[valid_time_mask]
        
        # Filter out NaN values in field data
        valid_data_mask = ~np.isnan(field_valid)
        field_no_nan = field_valid[valid_data_mask]
        time_no_nan = time_valid[valid_data_mask]
        
        # Filter out outliers using percentile-based method
        if len(field_no_nan) > 0:
            outlier_mask = filter_outliers(field_no_nan, lower_percentile=0.5, upper_percentile=99.5)
            field_plot = field_no_nan[outlier_mask]
            time_plot = time_no_nan[outlier_mask]
            
            # Count outliers for info
            num_outliers = np.sum(~outlier_mask)
            if num_outliers > 0:
                print(f"  {field_name}: Filtered out {num_outliers} outlier(s) ({100*num_outliers/len(field_no_nan):.1f}%)")
        else:
            field_plot = field_no_nan
            time_plot = time_no_nan
        
        # Use dodgerblue and cycle through nice colors
        colors = ['dodgerblue', 'steelblue', 'royalblue', 'cornflowerblue', 'deepskyblue',
                  'mediumblue', 'slateblue', 'mediumslateblue', 'lightsteelblue', 'lightskyblue']
        plot_color = colors[i % len(colors)]
        axes[i].plot(time_plot, field_plot, '-', linewidth=2, marker='o', markersize=3,
                    color=plot_color, alpha=0.8)
        axes[i].set_ylabel(field_name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        if i == num_fields - 1:
            axes[i].set_xlabel('Time Since Boot (seconds)', fontsize=12)
    
    plt.tight_layout()
    return fig


def create_overlay_plot(reader, fields, time_field='beac_time_since_boot',
                       packet_type='beacon', title='Telemetry Overlay Plot'):
    """
    Create an overlay plot of multiple telemetry fields on the same axes.
    
    Args:
        reader (TelemetryReader): The telemetry reader instance
        fields (list): List of field names to plot
        time_field (str): Name of the time field to use for x-axis
        packet_type (str): Packet type to filter by
        title (str): Title for the plot
    """
    # Filter by packet type
    filtered_data = reader.filter_by_packet_type(packet_type, case_sensitive=False)
    
    if len(filtered_data) == 0:
        print(f"Warning: No {packet_type} packets found. Cannot create plot.")
        return
    
    # Get available fields
    if hasattr(filtered_data, 'dtype') and hasattr(filtered_data.dtype, 'names'):
        available_fields = filtered_data.dtype.names
    else:
        # Columnar format
        available_fields = list(filtered_data.data_dict.keys())
    
    # Find time field
    time_fields = [f for f in available_fields if time_field.lower() in f.lower()]
    if not time_fields:
        print(f"Warning: Time field '{time_field}' not found.")
        return
    
    time_data = filtered_data[time_fields[0]]
    
    # Filter out NaN values in time
    valid_time_mask = ~np.isnan(time_data)
    time_valid = time_data[valid_time_mask]
    
    if len(time_valid) == 0:
        print("Warning: No valid time data found.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Since Boot (seconds)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot each field
    colors = plt.cm.tab10(np.linspace(0, 1, len(fields)))
    for i, field_name in enumerate(fields):
        if field_name not in available_fields:
            print(f"Warning: Field '{field_name}' not found. Skipping.")
            continue
        
        field_data = filtered_data[field_name]
        field_valid = field_data[valid_time_mask]
        
        # Filter out NaN values in field data
        valid_data_mask = ~np.isnan(field_valid)
        field_no_nan = field_valid[valid_data_mask]
        time_no_nan = time_valid[valid_data_mask]
        
        # Filter out outliers using percentile-based method
        if len(field_no_nan) > 0:
            outlier_mask = filter_outliers(field_no_nan, lower_percentile=0.5, upper_percentile=99.5)
            field_plot = field_no_nan[outlier_mask]
            time_plot = time_no_nan[outlier_mask]
            
            # Count outliers for info
            num_outliers = np.sum(~outlier_mask)
            if num_outliers > 0:
                print(f"  {field_name}: Filtered out {num_outliers} outlier(s) ({100*num_outliers/len(field_no_nan):.1f}%)")
        else:
            field_plot = field_no_nan
            time_plot = time_no_nan
        
        ax.plot(time_plot, field_plot, '-', linewidth=1.5, marker='o', markersize=2,
               label=field_name, color=colors[i])
    
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    return fig


def create_individual_plots(reader, packet_type='beacon', time_field='beac_time_since_boot',
                            output_dir=None, color='dodgerblue'):
    """
    Create individual plots for every variable in the specified packet type.
    
    Args:
        reader (TelemetryReader): The telemetry reader instance
        packet_type (str): Packet type to filter by (default: beacon)
        time_field (str): Name of the time field to use for x-axis
        output_dir (str): Directory to save plots (default: getenv('suncet_data')/trends/test_trends)
        color (str): Color to use for plots (default: dodgerblue)
    
    Returns:
        int: Number of plots created
    """
    # Filter by packet type
    filtered_data = reader.filter_by_packet_type(packet_type, case_sensitive=False)
    
    if len(filtered_data) == 0:
        print(f"Warning: No {packet_type} packets found. Cannot create plots.")
        return 0
    
    # Get available fields
    if hasattr(filtered_data, 'dtype') and hasattr(filtered_data.dtype, 'names'):
        available_fields = filtered_data.dtype.names
    else:
        # Columnar format
        available_fields = list(filtered_data.data_dict.keys())
    
    # Exclude time field and packet_type from plotting
    fields_to_plot = [f for f in available_fields 
                     if f not in [time_field, 'timestamp_seconds_since_boot', 'packet_type']]
    
    if len(fields_to_plot) == 0:
        print(f"Warning: No fields to plot for {packet_type} packets.")
        return 0
    
    # Find time field
    time_fields = [f for f in available_fields if time_field.lower() in f.lower()]
    if not time_fields:
        print(f"Warning: Time field '{time_field}' not found. Available fields: {available_fields[:10]}...")
        return 0
    
    time_data = filtered_data[time_fields[0]]
    
    # Filter out NaN values in time
    valid_time_mask = ~np.isnan(time_data)
    time_valid = time_data[valid_time_mask]
    
    if len(time_valid) == 0:
        print("Warning: No valid time data found.")
        return 0
    
    # Set up output directory
    if output_dir is None:
        suncet_data = os.getenv('suncet_data')
        if suncet_data:
            output_dir = os.path.join(suncet_data, 'trends', 'test_trends')
        else:
            output_dir = 'trends/test_trends'
            print(f"Warning: 'suncet_data' environment variable not set. Using default: {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define a nice color palette (dodgerblue and variations)
    colors = ['dodgerblue', 'steelblue', 'royalblue', 'cornflowerblue', 'deepskyblue',
              'mediumblue', 'slateblue', 'mediumslateblue', 'lightsteelblue', 'lightskyblue']
    
    num_plots = 0
    
    print(f"\nCreating individual plots for {len(fields_to_plot)} fields...")
    print(f"Output directory: {output_dir}")
    
    for i, field_name in enumerate(fields_to_plot):
        try:
            field_data = filtered_data[field_name]
            field_valid = field_data[valid_time_mask]
            
            # Filter out NaN values in field data
            valid_data_mask = ~np.isnan(field_valid)
            field_no_nan = field_valid[valid_data_mask]
            time_no_nan = time_valid[valid_data_mask]
            
            # Skip if no valid data
            if len(field_no_nan) == 0:
                print(f"  Skipping {field_name}: No valid data")
                continue
            
            # Filter out outliers using percentile-based method
            outlier_mask = filter_outliers(field_no_nan, lower_percentile=0.5, upper_percentile=99.5)
            field_plot = field_no_nan[outlier_mask]
            time_plot = time_no_nan[outlier_mask]
            
            # Skip if no data after filtering
            if len(field_plot) == 0:
                print(f"  Skipping {field_name}: No data after outlier filtering")
                continue
            
            # Create individual plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Use color from palette (cycle through if more fields than colors)
            plot_color = colors[i % len(colors)]
            
            ax.plot(time_plot, field_plot, '-', linewidth=2, marker='o', markersize=3,
                   color=plot_color, alpha=0.8)
            ax.set_xlabel('Time Since Boot (seconds)', fontsize=12)
            ax.set_ylabel(field_name, fontsize=12)
            ax.set_title(f'{field_name} vs Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot (sanitize filename)
            safe_filename = field_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            output_file = os.path.join(output_dir, f'{safe_filename}.png')
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            
            num_plots += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{len(fields_to_plot)} plots...")
        
        except Exception as e:
            print(f"  Error creating plot for {field_name}: {e}")
            continue
    
    print(f"\nSuccessfully created {num_plots} individual plots in {output_dir}")
    return num_plots


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Create plots of telemetry points from HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot default temperature stack plot from beacon packets
  python plot_telemetry.py
  
  # Plot specific fields
  python plot_telemetry.py --fields beac_batt_temp_1 beac_cdh_temp
  
  # Plot with custom file
  python plot_telemetry.py --filepath /path/to/file.h5
  
  # Create overlay plot instead of stack plot
  python plot_telemetry.py --overlay
        """
    )
    
    default_filepath = os.path.join(
        os.getenv('suncet_data', ''),
        'test_data',
        '2026-02-25_playback_test',
        '2026_056_15_35_11',
        'telemetry',
        'suncet_telemetry_mission_length_v1.0.1-hardline_playback_test.h5'
    )
    
    parser.add_argument('--filepath', type=str, default=default_filepath,
                       help='Path to telemetry HDF5 file')
    parser.add_argument('--fields', type=str, nargs='+',
                       help='List of field names to plot (if not specified, uses default temperature fields)')
    parser.add_argument('--packet-type', type=str, default='beacon',
                       help='Packet type to filter by (default: beacon)')
    parser.add_argument('--time-field', type=str, default='beac_time_since_boot',
                       help='Field name to use for x-axis time (default: beac_time_since_boot)')
    parser.add_argument('--overlay', action='store_true',
                       help='Create overlay plot instead of stack plot')
    parser.add_argument('--output', type=str,
                       help='Output filename for the plot (default: auto-generated)')
    parser.add_argument('--title', type=str,
                       help='Title for the plot')
    parser.add_argument('--individual', action='store_true',
                       help='Create individual plots for all fields in the packet type (saves to trends/test_trends)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for individual plots (default: getenv("suncet_data")/trends/test_trends)')
    
    args = parser.parse_args()
    
    # Read the telemetry file
    try:
        with TelemetryReader(args.filepath) as reader:
            # If --individual flag is set OR no fields specified, create individual plots for all fields
            if args.individual or (not args.fields and not args.overlay):
                num_plots = create_individual_plots(reader, 
                                                    packet_type=args.packet_type,
                                                    time_field=args.time_field,
                                                    output_dir=args.output_dir)
                if num_plots == 0:
                    print("Warning: No plots were created.")
                return
            
            # Determine which fields to plot (only if not creating individual plots)
            if args.fields:
                fields_to_plot = args.fields
                plot_title = args.title or 'Telemetry Plot'
            else:
                # Use default temperature fields
                print("No fields specified. Using default temperature fields from beacon packets...")
                available_fields = reader.get_field_names()
                default_keywords = get_default_temperature_fields()
                fields_to_plot = find_matching_fields(available_fields, default_keywords, case_sensitive=False)
                
                if not fields_to_plot:
                    print("Warning: No default temperature fields found. Available fields:")
                    print(f"  {available_fields[:20]}...")
                    return
                
                print(f"Found {len(fields_to_plot)} temperature fields to plot:")
                for field in fields_to_plot:
                    print(f"  - {field}")
                
                plot_title = args.title or 'Beacon Temperature Stack Plot'
            
            if not fields_to_plot:
                print("Error: No fields to plot.")
                return
            
            # Create the plot
            if args.overlay:
                fig = create_overlay_plot(reader, fields_to_plot, 
                                        time_field=args.time_field,
                                        packet_type=args.packet_type,
                                        title=plot_title)
            else:
                fig = create_stack_plot(reader, fields_to_plot,
                                       time_field=args.time_field,
                                       packet_type=args.packet_type,
                                       title=plot_title)
            
            if fig is None:
                print("Error: Failed to create plot.")
                return
            
            # Save the plot
            if args.output:
                output_file = args.output
            else:
                # Auto-generate filename
                if args.overlay:
                    output_file = 'telemetry_overlay_plot.png'
                else:
                    output_file = 'telemetry_stack_plot.png'
            
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_file}")
            plt.show()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
