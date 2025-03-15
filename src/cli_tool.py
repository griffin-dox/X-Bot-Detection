# CLI Tool for Batch Processing
import argparse

def main():
    parser = argparse.ArgumentParser(description='Bot Detection CLI')
    parser.add_argument('--text', type=str, help='Input text for bot detection')
    args = parser.parse_args()
    print(f'Input Text: {args.text}')

if __name__ == '__main__':
    main()