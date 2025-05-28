
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

  
</head>
<body>
    @include('components.alert')

    @if ($errors->any())
        <div style="color:red; margin-bottom: 10px;">
            <ul>
                @foreach ($errors->all() as $error)
                    <li>{{ $error }}</li>
                @endforeach
            </ul>
        </div>
    @endif

    <form action="{{ route('candidates.store') }}" method="post" enctype="multipart/form-data" class="m-5">
         @csrf

        <div>
            <label>Họ và tên</label><br>
            <input type="text" name="name" value="{{ old('name') }}">
        </div>

        <div>
            <label>Email</label><br>
            <input type="text" name="email" value="{{ old('email') }}">
        </div>

        <div>
            <label>Ngày sinh</label><br>
            <input type="date" name="birthday" value="{{ old('birthday') }}">
        </div>

        <div>
            <label>Ảnh đại diện</label><br>
            <input type="file" name="avatar" value="{{ old('avatar') }}">
        </div>

        <div>
            <label>CV </label><br>
            <input type="file" name="cv">
        </div>

        <div>
            <label>Mô tả ngắn</label><br>
            <input type="text" name="mota" >
        </div>
        <br>
        <button type="submit" class="btn btn-success">Gửi hồ sơ</button>
    </form>
    
</body>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

</html>
