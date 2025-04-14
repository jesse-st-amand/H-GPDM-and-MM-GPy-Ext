#!/usr/bin/env python
"""
Convert Excel files from MCCV analysis to LaTeX tables.

This script can be used to convert a specific Excel file or the most recent one
in a directory to a LaTeX table.

You can either:
1. Set the excel_path variable directly in the script
2. Run the script with command-line arguments
"""

import os
import argparse
from excel2latex import excel_to_latex, convert_latest_excel


def main():
    # =============================================================================
    # USER CONFIGURABLE SETTINGS
    # =============================================================================
    
    # SET YOUR EXCEL FILE PATH HERE - this will be used if no command-line arguments are provided
    # Use raw string (r"...") for Windows paths to handle backslashes correctly
    excel_path = r"C:\Users\Jesse\Documents\LaTex\GPDMM ICANN\sections\results\figures\data\BM_CMU_inter_data.xlsx"
    output_path = r"C:\Users\Jesse\Documents\LaTex\GPDMM ICANN\sections\results\figures\tables.tex"
    # TABLE CONTENT SETTINGS
    caption = "Model Comparisons for MCCV Analysis"  # Table caption
    label = "tab:model-comparison"                   # Label for cross-referencing
    
    # TABLE STYLE SETTINGS
    font_size = "tiny"            # Options: tiny, scriptsize, footnotesize, small, normalsize, large, Large
    resize_to_textwidth = True    # Whether to make table fit the page width
    use_booktabs = True           # Whether to use booktabs style (toprule, midrule, bottomrule)
    vertical_dividers = True      # Whether to include vertical lines between columns
    float_format = ".3f"          # Format for floating point numbers (e.g., ".3f" for 3 decimal places)
    include_hidden_columns = False # Whether to include columns that are hidden in Excel
    
    # =============================================================================
    # END OF USER SETTINGS
    # =============================================================================
    
    parser = argparse.ArgumentParser(description="Convert MCCV analysis Excel files to LaTeX tables")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()  # Not required anymore
    input_group.add_argument("--excel", help="Path to Excel file to convert")
    input_group.add_argument("--directory", help="Directory containing Excel files (to find most recent)")
    
    # Other options
    parser.add_argument("--output", help="Output file path (defaults to same name with .tex extension)")
    parser.add_argument("--caption", help="Table caption")
    parser.add_argument("--label", help="Table label for cross-referencing")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden columns")
    parser.add_argument("--font-size", help="LaTeX font size (tiny, scriptsize, footnotesize, small, normalsize)")
    parser.add_argument("--no-resize", action="store_true", help="Don't resize table to textwidth")
    parser.add_argument("--no-booktabs", action="store_true", help="Don't use booktabs style")
    parser.add_argument("--no-vertical-lines", action="store_true", help="Don't include vertical dividers")
    
    args = parser.parse_args()
    
    # Determine if using command-line args or direct path variable
    if args.excel or args.directory:
        # Command-line arguments provided - use those
        if args.excel:
            # Generate default caption and label if not provided
            if not args.caption:
                basename = os.path.basename(args.excel).replace('_', ' ').replace('.xlsx', '')
                args.caption = f"MCCV Analysis Results: {basename}"
            
            if not args.label:
                basename = os.path.basename(args.excel).replace('_', '-').replace('.xlsx', '')
                args.label = f"tab:mccv-{basename}"
                
            # Process other arguments
            use_font_size = args.font_size if args.font_size else font_size
            use_resize = not args.no_resize
            use_booktabs_style = not args.no_booktabs
            use_vertical_lines = not args.no_vertical_lines
                
            # Convert the file
            output_file = args.output or os.path.splitext(args.excel)[0] + '.tex'
            excel_to_latex(
                args.excel, 
                output_file, 
                caption=args.caption, 
                label=args.label,
                include_hidden_columns=args.include_hidden,
                font_size=use_font_size,
                resize_to_textwidth=use_resize,
                use_booktabs=use_booktabs_style,
                vertical_dividers=use_vertical_lines,
                float_format=float_format
            )
            print(f"Converted {args.excel} to {output_file}")
        
        elif args.directory:
            output_file = args.output
            try:
                use_font_size = args.font_size if args.font_size else font_size
                use_resize = not args.no_resize
                use_booktabs_style = not args.no_booktabs
                use_vertical_lines = not args.no_vertical_lines
                
                latex_file = convert_latest_excel(
                    args.directory, 
                    output_file, 
                    caption=args.caption or caption, 
                    label=args.label or label,
                    font_size=use_font_size,
                    resize_to_textwidth=use_resize,
                    use_booktabs=use_booktabs_style,
                    vertical_dividers=use_vertical_lines,
                    float_format=float_format,
                    include_hidden_columns=args.include_hidden
                )
                if latex_file:
                    print(f"Converted latest Excel file to {latex_file}")
            except FileNotFoundError as e:
                print(f"Error: {str(e)}")
                return 1
    else:
        # No command-line arguments - use the excel_path variable
        if os.path.exists(excel_path):
            # Use output_path if it's set, otherwise generate default path
            output_file = output_path if output_path else os.path.splitext(excel_path)[0] + '.tex'
            
            # Convert the file
            excel_to_latex(
                excel_path, 
                output_file, 
                caption=caption, 
                label=label,
                font_size=font_size,
                resize_to_textwidth=resize_to_textwidth,
                use_booktabs=use_booktabs,
                vertical_dividers=vertical_dividers,
                float_format=float_format,
                include_hidden_columns=include_hidden_columns
            )
            print(f"Converted {excel_path} to {output_file}")
        else:
            print(f"Error: Excel file not found at {excel_path}")
            print("Please update the excel_path variable or provide command-line arguments.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 