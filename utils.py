def out(output_writer, message, writer_new_line_char=True):
    print(message)
    if writer_new_line_char:
        output_writer.write(message + '\n')
    else:
        output_writer.write(message)
