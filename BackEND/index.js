var express = require("express");
var app = express();
var cors = require("cors");
const path = require("path"); // Import path module
const bodyParser = require("body-parser");
app.use(cors());
const router = express.Router();
const fs = require('fs');
const moment = require('moment-timezone');

// Define your routes here
router.get('/', (req, res) => {
    res.send('Hello from index.js');
});


app.use(express.static(path.join(__dirname, 'FrontEND')));



//here is my grade.js file

const multer = require('multer');
const xlsx = require('xlsx');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static('public'));  // Serve static files from public directory


const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        // Ensure the 'uploads' directory exists or is created
        const uploadDir = 'Uploads/';
        fs.mkdirSync(uploadDir, { recursive: true });

        let dest;
        const studentId = req.body.studentId; // Assuming studentId is part of the request body

        // Determine the destination directory based on file type
        if (file.mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
            // Grading excel sheets
            dest = `${uploadDir}`;
        } else {
            // Scholarship documents
            dest = `Uploads/ScholarshipUploads/${studentId}/`;
            fs.mkdirSync(dest, { recursive: true });
        }

        // Callback with the destination directory
        cb(null, dest);
    },
    filename: function (req, file, cb) {
        const filename = `${Date.now()}-${file.originalname}`;
        cb(null, filename);
    }
});
const upload = multer({ storage: storage });
app.get('/ScholarshipApplications', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/ScholarshipApplications.html'));
});

// Endpoint to handle the scholarship form submission
app.post('/submit-scholarship', upload.fields([
    { name: 'salaryCertificate', maxCount: 1 },
    { name: 'itr', maxCount: 1 },
    { name: 'incomeCertificate', maxCount: 1 },
    { name: 'incomeAffidavit', maxCount: 1 }
]), (req, res) => {
    console.log('Files uploaded:', req.files);
    res.send("Scholarship application submitted successfully!");
});

app.use('/uploads', express.static(path.join(__dirname, 'uploads', 'ScholarshipUploads'), {
    setHeaders: function (res, path) {
        console.log("Serving:", path); // This will log the physical path attempts to access
    }
}));

// New route to fetch all scholarship applications
app.get('/applications', (req, res) => {
    const applicationsPath = path.join(__dirname, 'uploads', 'ScholarshipUploads');
    fs.readdir(applicationsPath, (err, folders) => {
        if (err) {
            console.error(err);
            return res.status(500).send('Error retrieving applications.');
        }
        let applications = [];
        folders.forEach(folder => {
            let files = fs.readdirSync(path.join(applicationsPath, folder));
            applications.push({ studentId: folder, files: files });
        });
        res.json(applications);
    });
});

// Serve files from the 'uploads' directory, removing the case sensitivity issue
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

app.use(express.static('public'));

app.post('/upload', upload.single('file'), (req, res) => {
    if (req.file) {
        readFile(req.file.buffer).then(rows => {
            const data = processData(rows);
            res.json(data);
        }).catch(error => {
            res.status(500).send("Failed to read the Excel file.");
        });
    } else {
        res.status(400).send("No file uploaded.");
    }
});

function processData(rows) {
    // Extract and process data from rows
    return {
        // processed data
    };
}


const corsOptions = {
    origin: 'http://127.0.0.1:5500', // Allow requests from this origin
    methods: 'POST', // Allow only POST requests
    optionsSuccessStatus: 200 // Respond with 200 for preflight requests
};

app.use(cors(corsOptions));




app.post('/submit-grades', upload.single('gradesFile'), (req, res) => {
    const file = req.file;
    const courseCode = req.body.courseCode;

    if (!file) {
        return res.status(400).send('No file uploaded.');
    }

    try {
        const workbook = xlsx.readFile(file.path);
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const gradesData = xlsx.utils.sheet_to_json(sheet);

        if (!gradesData.length) {
            return res.status(400).send('No data found in the file.');
        }

        const sql = 'SELECT Student_ID FROM course_enrollment WHERE Course_Code = ?';
        pool.query(sql, [courseCode], (error, results) => {
            if (error) {
                console.error('Database error:', error);
                return res.status(500).send('Database error: Unable to fetch registered students.');
            }

            if (results.length === 0) {
                return res.status(404).send('No students registered for this course.');
            }

            const registeredStudents = results.map(row => row.Student_ID);
            const validGradesData = gradesData.filter(entry => registeredStudents.includes(entry.Student_ID));

            if (validGradesData.length === 0) {
                return res.status(404).send('No valid grades found to update for registered students.');
            }

            const updateSql = 'INSERT INTO grades (Student_ID, Course_Code, Grade) VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE Grade = VALUES(Grade)';
            validGradesData.forEach((entry, index) => {
                pool.query(updateSql, [entry.Student_ID, courseCode, entry.Grade], (error) => {
                    if (error) {
                        console.error('Error updating grade for student ID:', entry.Student_ID, error);
                    }
                    if (index === validGradesData.length - 1) {
                        res.send('Grades submitted successfully.');
                    }
                });
            });
        });
    } catch (err) {
        console.error('Error processing file:', err);
        return res.status(500).send('Failed to process file.');
    }
});


//add students code 

app.post('/add-students', upload.single('studentsFile'), async (req, res) => {
    const file = req.file;

    if (!file) {
        return res.status(400).send('No file uploaded.');
    }

    try {
        const workbook = xlsx.readFile(file.path);
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const studentsData = xlsx.utils.sheet_to_json(sheet);

        if (!studentsData.length) {
            return res.status(400).send('No data found in the file.');
        }

        const results = {
            success: [],
            errors: []
        };

        for (const student of studentsData) {
            const { Student_ID, Student_Name, Branch, Batch, Gender, Email, CGPA, Faculty_Advisor_1, Instructor_ID } = student;
            const insertSql = 'INSERT INTO student (Student_ID, Student_Name, Branch, Batch, Gender, Email, CGPA, Faculty_Advisor, Instructor_ID) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)';

            try {
                await pool.query(insertSql, [Student_ID, Student_Name, Branch, Batch, Gender, Email, CGPA, Faculty_Advisor_1, Instructor_ID]);
                results.success.push(`Student ${Student_ID} - ${Student_Name} added successfully.`);
            } catch (error) {
                if (error.code === 'ER_DUP_ENTRY') {
                    results.errors.push(`Duplicate entry for Student_ID ${Student_ID}.`);
                } else {
                    results.errors.push(`Error adding Student_ID ${Student_ID}: ${error.sqlMessage}`);
                }
            }
        }

        res.json(results);
    } catch (err) {
        console.error('Error processing file:', err);
        return res.status(500).send('Failed to process file.');
    }
});
// Export the router
module.exports = router;

const pool = require('./db'); // Import the pool variable

app.use(express.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.json());




app.get('/contact', function (req, res) {
    res.render('contact', { qs: req.query });
});

app.get("/", (req, res) => {
    res.send("working");
});


app.get("/student", (req, res) => {
    pool.query('SELECT * FROM student', (err, results, fields) => {
        if (err) {
            console.error('Error fetching student data:', err);
            res.status(500).send("Internal Server Error");
        } else {
            res.json(results);
        }
    });
});

app.get("/courses", (req, res) => {
    pool.query('SELECT * FROM courses', (err, results, fields) => {
        if (err) {
            console.error('Error fetching course data:', err);
            res.status(500).send("Internal Server Error");
        } else {
            res.json(results);
        }
    });
});

app.get("/view-courses", (req, res) => {
    const userEmail = req.query.username; // Assuming email is passed as a query parameter

    console.log(`Fetching courses for user ${userEmail}`);

    // Execute MySQL query to select courses associated with the provided email
    pool.query('SELECT * FROM courses WHERE Email = ?', [userEmail], (err, results, fields) => {
        if (err) {
            console.error('Error fetching course data:', err);
            res.status(500).send("Internal Server Error");
        } else {
            res.json(results);
            console.log(results);
        }
    });
});

app.get("/search", (req, res) => {
    const query = req.query.query;
    const searchQuery = '%' + query + '%';
    const sql = `SELECT * FROM student WHERE Student_Name LIKE ? OR Student_ID LIKE ? OR Branch LIKE ? OR Batch LIKE ? OR Gender LIKE ? OR Email LIKE ?`;
    const values = [searchQuery, searchQuery, searchQuery, searchQuery, searchQuery, searchQuery];

    pool.query(sql, values, (err, studentResults) => {
        if (err) {
            console.error("Error fetching student data:", err);
            return res.status(500).send("Internal Server Error");
        }

        pool.query(`SELECT * FROM courses WHERE Course_Code LIKE ? OR Instructor_Name LIKE ? OR School LIKE ? OR Email LIKE ?`, values, (err, courseResults) => {
            if (err) {
                console.error("Error fetching course data:", err);
                return res.status(500).send("Internal Server Error");
            }

            const results = {
                students: studentResults,
                courses: courseResults
            };
            res.json(results);
        });
    });
});

app.post("/submit-form", (req, res) => {
    const { studentID, studentName, branch, batch, gender, email, cgpa, Instructor_ID } = req.body;

    // First, fetch the instructor name using the Instructor_ID
    pool.query('SELECT Instructor_Name FROM Instructor WHERE Instructor_ID = ?', [Instructor_ID], (err, result) => {
        if (err) {
            console.error('Error fetching instructor:', err);
            return res.status(500).send("Internal Server Error");
        }
        if (result.length === 0) {
            return res.status(404).send("Faculty Advisor not found. Please check the ID and try again.");
        }
        const Faculty_Advisor = result[0].Instructor_Name;

        // Then, proceed to insert the new student with the fetched Faculty_Advisor
        pool.query('INSERT INTO student (Student_ID, Student_Name, Branch, Batch, Gender, Email, CGPA, Faculty_Advisor, Instructor_ID) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [studentID, studentName, branch, batch, gender, email, cgpa, Faculty_Advisor, Instructor_ID],
            (err, results) => {
                if (err) {
                    console.error('Error adding student:', err);
                    return res.status(500).send("Internal Server Error");
                }
                console.log('Student added successfully');
                res.json({ message: 'Student added successfully' });
            });
    });
});


app.post("/submit-course", (req, res) => {
    const { courseCode, school, instructorName, email } = req.body;

    pool.query('INSERT INTO courses (Course_Code, School, Instructor_Name, Email) VALUES (?, ?, ?, ?)',
        [courseCode, school, instructorName, email],
        (err, results) => {
            if (err) {
                console.error('Error adding course:', err);
                return res.status(500).send("Internal Server Error");
            }
            console.log('Course added successfully');
            res.json({ message: 'Course added successfully' });
        });
});

app.get('/enrolled-students', (req, res) => {
    const courseCode = req.query.course;

    const query = `SELECT student.Student_ID, student.Student_Name, student.Branch, student.Batch, student.Email
        FROM student
        INNER JOIN course_enrollment ON student.Student_ID = course_enrollment.Student_ID
        WHERE course_enrollment.Course_code = ?`;

    pool.query(query, [courseCode], (err, results) => {
        if (err) {
            console.error('Error fetching enrolled students:', err);
            res.status(500).send('Error fetching enrolled students.');
            return;
        }
        res.json(results);
    });
});

app.get('/filter', (req, res) => {
    const { branch, batch, gender, cgpaMin, cgpaMax } = req.query;

    let sql = 'SELECT * FROM student WHERE 1=1';
    const values = [];

    if (branch) {
        sql += ' AND Branch = ?';
        values.push(branch);
    }
    if (batch) {
        sql += ' AND Batch = ?';
        values.push(batch);
    }
    if (gender) {
        sql += ' AND Gender = ?';
        values.push(gender);
    }
    if (cgpaMin && cgpaMax) {
        sql += ' AND CGPA BETWEEN ? AND ?';
        values.push(parseFloat(cgpaMin), parseFloat(cgpaMax));
    }

    pool.query(sql, values, (err, results) => {
        if (err) {
            console.error('Error executing query:', err);
            res.status(500).json({ error: 'Internal server error' });
            return;
        }
        res.json(results);
    });
});

app.post('/login', (req, res) => {
    const username = req.body.username;
    const password = req.body.password;
    const userType = req.body.userType;

    const sql = 'SELECT * FROM users WHERE username = ? AND password_hash = ?';
    pool.query(sql, [username, password], (err, results) => {
        if (err) {
            console.error('Error querying database: ' + err);
            return res.status(500).json({ error: 'Internal Server Error' });
        }

        if (results.length === 0) {
            // No user found with the provided credentials
            return res.status(401).json({ error: 'Unknown Username' });
        }

        const user = results[0];
        // Check if the user's role matches the provided userType
        if (user.role !== userType) {
            return res.status(401).json({ error: 'Incorrect User Type' });
        }

        // If everything is correct, return success
        res.json({ success: true, user: user });
    });
});



app.get('/student/:username', (req, res) => {
    const username = req.params.username;

    const query = `SELECT * FROM student_profile WHERE Student_ID = ?`;
    pool.query(query, [username], (err, results) => {
        if (err) {
            console.error('Error querying MySQL:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        if (results.length === 0) {
            return res.status(404).json({ error: 'Student not found' });
        }

        // Extract student details from the query results
        const studentDetails = results[0];
        res.json(studentDetails);
    });
});





app.get('/get-profile', (req, res) => {
    const sql = "SELECT sp.*, sa.Faculty_Advisor AS Faculty_Advisor FROM student_profile sp JOIN student sa ON sp.Student_ID = sa.Student_ID WHERE sp.Student_ID = ?;";

    const username = req.query.username; // Change parameter name to 'username'

    pool.query(sql, [username], (err, result) => {
        if (err) {
            console.error('Error executing SQL query:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        if (result.length === 0) {
            return res.status(404).json({ error: 'Student not found' });
        }
        console.log(result[0]);
        res.json(result[0]);
    });
});



app.post('/updateProfile', (req, res) => {
    const { Student_ID, Address, Father_Mobile } = req.body;
    const sql = "UPDATE student_profile SET Address = ?,Father_Mobile=? WHERE Student_ID = ?";
    pool.query(sql, [Address, Father_Mobile, Student_ID], (err, result) => {
        if (err) {
            console.error('Error updating profile:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        res.json({ message: 'Profile updated successfully' });
    });
});

app.post('/get_courses_with_grades', (req, res) => {
    const studentId = req.body.studentId;

    // Corrected SQL query (removed the extra comma)
    const sql = `
        SELECT 
            courses.Course_Code, 
            courses.Instructor_Name, 
            courses.Course_Type, 
            course_enrollment.Semester AS Enrollment_Semester,
            courses.Credit, 
            grades.Grade
        FROM 
            courses     
        INNER JOIN 
            course_enrollment ON courses.Course_Code = course_enrollment.Course_Code 
        LEFT JOIN 
            grades ON courses.Course_Code = grades.course_code AND course_enrollment.Student_ID = grades.student_id
        WHERE 
            course_enrollment.Student_ID = ?`;

    // Use pool.query to execute SQL queries
    pool.query(sql, [studentId], (err, results) => {
        if (err) {
            console.error('Error fetching courses with grades:', err);
            res.status(500).json({ error: 'Internal server error' });
            return;
        }
        console.log(results);
        res.json(results);
    });
});



app.get("/get-reg-courses", (req, res) => {
    const { Department, Semester } = req.query;
    const query = `SELECT Course_Code, Instructor_Name, Email, Course_Type,Prerequisites FROM courses WHERE Department LIKE '%${Department}%' AND Semester LIKE '%${Semester}%'`;
    pool.query(query, (err, results) => {
        if (err) {
            console.error("Error fetching courses:", err);
            res.status(500).json({ error: "Internal server error" });
            return;
        }
        console.log(results);
        res.json({ courses: results });
    });
});


app.get('/course_reg', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/course_reg.html'));
});

app.get('/profile', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/profile.html'));
});

app.get('/add-instructor', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/add-instructor.html'));
});
app.get('/adminRegistration', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/adminRegistration.html'));
});
app.get('/CourseBySem', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/CourseBySem.html'));
});

app.get('/StudRegRequests', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/StudRegRequests.html'));
});
app.get('/Scholarship', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'FrontEND', './Components/Scholarship.html'));
});


///only for submit-course-reg
const mysql1 = require('mysql2/promise');
const { Console } = require("console");
const pool1 = mysql1.createPool({
    connectionLimit: 10,
    host: 'localhost',
    database: 'College_db',
    user: 'root',
    password: '6350'
});


// run these on terminal:-  npm install nodemailer

const nodemailer = require('nodemailer');

let transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'sdms4coursereg@gmail.com',
        pass: 'sgmgynevndglrwip'
    }
});





app.post('/submit-course-reg', async (req, res) => {
    const { studentId, courses, semesterNumber } = req.body;

    if (!Array.isArray(courses) || courses.length === 0) {
        return res.status(400).json({ error: 'Invalid request: courses must be an array and cannot be empty' });
    }

    try {
        const alreadyRegistered = [];
        for (const courseCode of courses) {
            // Check for previous registrations in the Registration table
            const [existingInRegistration] = await pool1.query(
                'SELECT Course_Code FROM Registration WHERE Student_ID = ? AND Course_Code = ?',
                [studentId, courseCode]
            );

            if (existingInRegistration.length > 0) {
                alreadyRegistered.push(courseCode);
                continue;  // Skip further checks if already registered
            }

            // Check for enrollment in the course_enrollment table
            const [existingInEnrollment] = await pool1.query(
                'SELECT Course_Code FROM course_enrollment WHERE Student_ID = ? AND Course_Code = ?',
                [studentId, courseCode]
            );

            if (existingInEnrollment.length > 0) {
                alreadyRegistered.push(courseCode);
                continue;  // Skip further checks if already enrolled
            }

            // Check prerequisites
            const prerequisites = await pool1.query(
                'SELECT Prerequisites FROM courses WHERE Course_Code = ?',
                [courseCode]
            );

            // Only check prerequisites if they exist
            if (prerequisites[0].length > 0 && prerequisites[0][0].Prerequisites) {
                for (const prereq of prerequisites[0]) {
                    const grades = await pool1.query(
                        `SELECT Grade FROM grades WHERE Student_ID = ? AND Course_Code = ? AND Grade NOT IN ('E', 'F')`,
                        [studentId, prereq.Prerequisites]
                    );

                    if (grades[0].length === 0) {
                        // Prerequisite not satisfied
                        return res.status(402).json({
                            error: `Prerequisite not satisfied for course ${courseCode}.`,
                            message: `Prerequisite ${prereq.Prerequisites} not satisfied for course ${courseCode}.`
                        });
                    }
                }
            }

            // Checking if the instructor is associated with the student
            const [[studentData]] = await pool1.query('SELECT Instructor_ID FROM student WHERE Student_ID = ?', [studentId]);
            if (!studentData || !studentData.Instructor_ID) {
                return res.status(404).json({ error: `No instructor found for student with ID ${studentId}` });
            }

            const instructorId = studentData.Instructor_ID;



            // Insert the course into Registration table after passing all checks
            await pool1.query(
                'INSERT INTO Registration (Student_ID, Course_Code, Instructor_ID, Semester) VALUES (?, ?, ?, ?)',
                [studentId, courseCode, instructorId, semesterNumber]
            );
            // Send email to the instructor
            const instructorEmail = await pool1.query('SELECT Instructor_Email FROM Instructor WHERE Instructor_ID = ?', [instructorId]);
            console.log(instructorEmail[0][0].Instructor_Email);
            if (instructorEmail[0].length > 0) {
                let mailOptions = {
                    from: 'sdms4coursereg@gmail.com',
                    // to: instructorEmail[0][0].Instructor_Email,
                    to: 'rajesh.meena.20033@iitgoa.ac.in',
                    subject: `New Course Registration by ${studentId}`,
                    text: `A  student with ID ${studentId} has registered for course ${courseCode}. Please Review It.`
                };

                transporter.sendMail(mailOptions, function (error, info) {
                    if (error) {
                        console.log('Error sending email:', error);
                    } else {
                        console.log('Email sent: ' + info.response);
                    }
                });
            }
        }


        if (alreadyRegistered.length > 0) {
            res.status(400).json({
                message: "Some courses are already registered : ",
                courses: alreadyRegistered
            });
        } else {
            res.json({ message: 'All courses registered successfully' });
        }
    } catch (error) {
        console.error('Error processing registration:', error);
        res.status(500).json({ error: 'Failed to process registration' });
    }
});


// Endpoint to fetch the list of instructors
app.get("/instructor", (req, res) => {
    // Execute MySQL query to select all instructors from the database
    pool.query('SELECT * FROM Instructor', (err, results) => {
        if (err) {
            console.error('Error fetching instructors:', err);
            return res.status(500).send("Internal Server Error");
        }
        res.json(results); // Send the list of instructors as JSON response
    });
});

app.post("/submit-instructor", (req, res) => {
    const { instructorName, instructorEmail, school } = req.body;

    // Execute MySQL query to insert instructor data into the database
    pool.query('INSERT INTO Instructor (instructor_Name, instructor_Email, School) VALUES (?, ?, ?)',
        [instructorName, instructorEmail, school],
        (err, results) => {
            if (err) {
                console.error('Error adding instructor:', err);
                return res.status(500).send("Internal Server Error");
            }
            console.log('instructor added successfully');
            res.json({ message: 'instructor added successfully' });
        });
});





app.get('/student/:id', (req, res) => {
    const studentID = req.params.id;

    // Query to fetch student data from the database
    const query = `SELECT * FROM student WHERE Student_ID = ?`;

    // Execute the query using MySQL pool
    pool.query(query, [studentID], (error, results) => {
        if (error) {
            console.error('Error fetching student data:', error);
            res.status(500).json({ error: 'Internal server error' });
            return;
        }


        if (results.length === 0) {
            res.status(404).json({ error: 'Student not found' });
            return;
        }

        // Send student data as JSON response
        res.json(results[0]);
    });
});

app.get('/get-student-profile', (req, res) => {
    // Extract studentId from the query parameters
    const studentId = req.query.studentId;

    // SQL query to retrieve the student's profile data based on their Student_ID
    const sql = 'SELECT Branch FROM student_profile WHERE Student_ID = ?';

    // Execute the SQL query
    pool.query(sql, [studentId], (error, results) => {
        if (error) {
            console.error('Error fetching student profile:', error);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            if (results.length > 0) {
                // Extract department (Branch) from the query results
                const department = results[0].Branch;

                // Send the department data as JSON response
                res.json({ department });
            } else {
                res.status(404).json({ error: 'Student profile not found' });
            }
        }
    });
});


app.get('/get-registered-courses-by-sem', (req, res) => {
    const { semester, studentId } = req.query;
    if (!semester || !studentId) {
        return res.status(400).send('Semester and student ID are required');
    }

    const sql = `
        SELECT c.Course_Code, c.Instructor_Name, c.Course_Type, c.Credit, g.Grade
        FROM courses c
        JOIN grades g ON c.Course_Code = g.Course_Code
        WHERE c.Semester = ? AND g.Student_ID = ?
    `;
    console.log(semester, studentId);
    pool.query(sql, [semester, studentId], (err, results) => {
        if (err) {
            console.error('Failed to fetch courses:', err);
            return res.status(500).send('Failed to fetch courses');
        }
        console.log(results);
        res.json(results);
    });
});

// Endpoint to check registration status for a semester
app.get('/check-registration-status', (req, res) => {
    const { semester } = req.query;
    if (!semester) {
        return res.status(400).send('Semester is required');
    }

    const sql = 'SELECT open_date, close_date FROM registration_periods WHERE semester = ?';

    pool.query(sql, [semester], (err, results) => {
        if (err) {
            console.error('Failed to check registration status:', err);
            return res.status(500).send('Failed to fetch registration period');
        }
        console.log("Raw database results:", results);
        if (results.length === 0 || !results[0].open_date || !results[0].close_date) {
            // Treat as closed if there are no entries or missing time values
            return res.json({ isOpen: false });
        }

        const { open_date, close_date } = results[0];

        const now = new Date(); // Ensure 'now' is defined
        const isOpen = now >= new Date(open_date) && now <= new Date(close_date);

        res.json({ isOpen });
    });
});


app.post('/set-registration-dates', (req, res) => {
    console.log(req.body); // Log the incoming request body to debug

    const { semester, openDate, closeDate } = req.body;
    if (!semester || !openDate || !closeDate) {
        return res.status(400).json({ message: "All fields are required" });
    }

    const sql = `INSERT INTO registration_periods (semester, open_date, close_date)
                 VALUES (?, ?, ?)
                 ON DUPLICATE KEY UPDATE open_date = VALUES(open_date), close_date = VALUES(close_date)`;

    pool.query(sql, [semester, openDate, closeDate], (err, result) => {
        if (err) {
            console.error('Failed to set registration dates:', err);
            return res.status(500).json({ message: "Database error", error: err.message });
        }
        res.json({ message: "Registration dates successfully updated" });
    });
});
app.post('/end-registration-now', (req, res) => {
    const { semester } = req.body;

    const closeDate = moment.utc().tz('Asia/Kolkata').format('YYYY-MM-DD HH:mm:ss');
    console.log(semester, closeDate);
    const sql = `UPDATE registration_periods SET close_date = ? WHERE semester = ?`;
    console.log(sql);
    pool.query(sql, [closeDate, semester], (error, results) => {
        if (error) {
            console.error('Error ending registration:', error);
            return res.status(500).json({ message: 'Error ending registration', error });
        }
        res.json({ message: 'Registration ended and date updated successfully', results });
    });
});





app.post('/update-registration-status', async (req, res) => {
    const { studentId, courseCode, status } = req.body;

    console.log('Updating registration status to:', status, 'for studentId:', studentId, 'and courseCode:', courseCode);

    try {
        const updateResult = await pool1.query(
            'UPDATE Registration SET Registration_Status = ? WHERE Student_ID = ? AND Course_Code = ?',
            [status, studentId, courseCode]
        );
        console.log(updateResult[0].affectedRows);


        // Check if the update operation was successful and the status is "Approved"
        if (status === 'Approved') {

            // First, check if the course is already enrolled
            const [existingEnrollment] = await pool1.query(
                'SELECT * FROM course_enrollment WHERE Student_ID = ? AND Course_Code = ?',
                [studentId, courseCode]
            );

            if (existingEnrollment.length > 0) {
                // If already enrolled, send a message back without inserting
                return res.status(409).json({ message: 'Course already enrolled' });
            }

            // If not enrolled, proceed to get the semester and enroll
            const [semesterResult] = await pool1.query(
                'SELECT Semester FROM Registration WHERE Student_ID = ? AND Course_Code = ?',
                [studentId, courseCode]
            );

            if (semesterResult.length === 0) {
                // Handle unexpected case where semester data is missing
                return res.status(404).json({ error: 'Semester information not found for registration' });
            }

            const semester = semesterResult[0].Semester;

            // Proceed to insert into course_enrollment
            const insertResult = await pool1.query(
                'INSERT INTO course_enrollment (Student_ID, Course_Code, Semester) VALUES (?, ?, ?)',
                [studentId, courseCode, semester]
            );

            if (insertResult[0].affectedRows > 0) {
                console.log('Course enrollment recorded successfully for student:', studentId);
                res.json({ message: 'Registration status updated and course enrolled successfully' });
            } else {
                // Handle unexpected case where no row was inserted
                throw new Error('Failed to enroll course despite successful status update');
            }
        } else if (updateResult[0].affectedRows > 0) {
            // If the status is not "Approved" but was updated
            res.json({ message: 'Registration status updated successfully' });
        } else {
            // If no rows were updated
            res.status(404).json({ error: 'No such registration found or no update needed' });
        }
    } catch (error) {
        console.error('Database error:', error);
        res.status(500).json({ error: "Failed to update registration status or enroll course" });
    }
});


app.get('/get-registration-requests', (req, res) => {
    const { studentId } = req.query;  // Access studentId from query parameters

    if (!studentId) {
        return res.status(400).json({ error: 'Student ID is required' });
    }

    const query = `
        SELECT Course_Code, Registration_Status,Student_ID,Semester
        FROM Registration
        WHERE Student_ID = ?
    `;

    pool.query(query, [studentId], (err, results) => {
        if (err) {
            console.error('Error fetching registration requests:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        if (results.length === 0) {
            return res.status(404).json({ message: 'No registration requests found for this student' });
        }
        console.log(results);
        res.json(results);
    });
});

// Get Instructor ID by email
app.get('/get-instructor-id/:email', (req, res) => {
    const { email } = req.params;

    // SQL query to fetch the instructor ID
    const query = 'SELECT Instructor_ID FROM Instructor WHERE Instructor_Email = ?';
    console.log("sdfd");
    // Execute the query using the email parameter
    pool.query(query, [email], (err, results) => {
        if (err) {
            console.error('Error executing query:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }

        if (results.length === 0) {
            return res.status(404).json({ error: 'Instructor not found' });
        }

        // Return the instructor ID
        console.log(results[0].Instructor_ID);
        res.json({ id: results[0].Instructor_ID });

    });
});


app.get('/get-FAs-students', (req, res) => {
    const instructorId = req.query.instructorId;
    if (!instructorId) {
        return res.status(400).json({ error: "Instructor ID is required" });
    }

    const sql = `
        SELECT s.Student_ID, s.Student_Name, s.Faculty_Advisor
        FROM student s
        WHERE s.Instructor_ID = ?
    `;

    pool.query(sql, [instructorId], (err, results) => {
        if (err) {
            console.error('Error fetching students:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        if (results.length === 0) {
            return res.status(404).json({ message: 'No students found for this instructor' });
        }
        res.json(results);
    });
});


// Endpoint to get all courses with grades for a student
app.get('/get-student-courses', (req, res) => {
    const { studentId } = req.query;

    if (!studentId) {
        return res.status(400).json({ error: 'Student ID is required' });
    }

    const query = `
        SELECT 
            c.Course_Code, 
            c.School, 
            c.Instructor_Name, 
            ce.Semester,
            c.Credit, 
            c.Course_Type,
            g.Grade
        FROM 
            courses c
        JOIN 
            course_enrollment ce ON c.Course_Code = ce.Course_Code
        LEFT JOIN 
            grades g ON g.Course_Code = c.Course_Code AND g.Student_ID = ce.Student_ID
        WHERE 
            ce.Student_ID = ?
    `;

    pool.query(query, [studentId], (err, results) => {
        if (err) {
            console.error('Error fetching courses:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        if (results.length === 0) {
            return res.status(404).json({ message: 'No courses found for this student' });
        }
        res.json(results);
    });
});


app.get('/get-all-students', (req, res) => {
    const sql = 'SELECT * FROM student'; // Adjust SQL as needed
    pool.query(sql, (err, results) => {
        if (err) {
            console.error('Failed to fetch students:', err);
            res.status(500).send('Failed to fetch students');
        } else {
            res.json(results);
        }
    });
});

// 

app.get("/get-registrations/:studentId", (req, res) => {
    const studentId = parseInt(req.params.studentId);
    const query = `
        SELECT Student_ID, Course_Code, Instructor_ID, Semester, Registration_Status, Registration_Date 
        FROM Registration 
        WHERE Student_ID = ?;
    `;

    pool.query(query, [studentId], (err, results) => {
        if (err) {
            console.error("Error fetching registrations:", err);
            res.status(500).json({ error: "Internal server error" });
            return;
        }
        // console.log(results);
        res.json(results);
    });
});

app.post("/cancel-registration", (req, res) => {
    const { studentId, courseId } = req.body;

    pool.query('BEGIN', (err) => {
        if (err) {
            console.error("Error starting transaction:", err);
            res.status(500).json({ error: "Internal server error starting transaction" });
            return;
        }

        const updateQuery = `UPDATE Registration SET Registration_Status = 'Cancelled' WHERE Student_ID = ? AND Course_Code = ? AND Registration_Status != 'Cancelled'`;
        pool.query(updateQuery, [studentId, courseId], (err, updateResult) => {
            if (err || updateResult.rowCount === 0) {
                pool.query('ROLLBACK', (err) => {
                    if (err) {
                        console.error("Error rolling back transaction:", err);
                    }
                    res.status(404).json({ message: "Registration not found or already cancelled" });
                });
                return;
            }

            const deleteQuery = `DELETE FROM course_enrollment WHERE Student_ID = ? AND Course_Code = ? `;
            pool.query(deleteQuery, [studentId, courseId], (err) => {
                console.log("AFFD",);
                if (err) {
                    pool.query('ROLLBACK', (err) => {
                        if (err) {
                            console.error("Error rolling back transaction:", err);
                        }
                        res.status(500).json({ error: "Internal server error during deletion" });
                    });
                    return;
                }

                pool.query('COMMIT', (err) => {
                    if (err) {
                        console.error("Error committing transaction:", err);
                        res.status(500).json({ error: "Internal server error during commit" });
                        return;
                    }
                    res.json({ message: "Registration cancelled and enrollment deleted successfully" });
                });
            });
        });
    });
});

var port = process.env.PORT || 5000
app.listen(port, function (req, res) {
    console.log(`server started at port ${port}`);
});
