@foreach ($posts as $post)
    <div>
        <h3>{{ $post->title }}</h3>
        <p>{{ $post->content }}</p>

        @can('edit-post', $post)
            <a href="{{ route('posts.edit', $post->id) }}">Sửa bài viết</a>
        @endcan
    </div>
@endforeach
