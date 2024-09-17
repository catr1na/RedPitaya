
int acquire_init(int num_segments, int segment_length, int decimation,float trigger_level);

void acquire_read_and_transform();

void acquire_read_and_transform_select();

void acquire_apply_template();

int acquire_write_out(char* filename);

int acquire_write_out_template_buff(char* filename);

int acquire_write_out_select(char* filename);

int acquire_clean();

float get_select_buffer_at(int i);

float select_buffer_max();

int check_trigger();
